# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 08:42:42 2025
Flux Quantification Code 
@author: ppena

----------PLEASE READ----------

DESCRIPTION
This script takes in methane concentration data and applies a series of user defined functions to calculate steady state and transient responses for flux.
It then outputs a Report with Figures comparing transient and steady state flux responses.
 
 INPUT: 
     Data, Methane Concentrations .csv file
     
     USER INPUTS:
         Start_Time: Start of visualization
         
         End_Time: End of visualization
 
 OUTPUT:
     
     Plot of average box concentation saved in /Figures as Avg Box Concentation"+date+".png
     
     Plot of average ambient concentation saved in /Figures as Avg Box Concentation"+date+".png
     
     Plot of transient and steady state flux rates saved in /Figures as Flux Rates"+date+".png
     
     Report of Flux Quantification saved in /Reports as FluxReport
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from fpdf import FPDF
import tkinter as tk
from tkinter import filedialog

##################INPUTS#################
Start_Time = pd.to_datetime("2025-04-23 14:30:00")  # start time
End_Time = pd.to_datetime("2025-04-23 18:45:00")  # end time
#########################################
# %% User Defined Functions

# Converting Celcius to Kelvin


def CtoK(C):
    K = C + 273.15
    return K

#Volume [m^3]


def Volume(height, area):
    Vol = height*area
    return Vol

# Converting [ppm] to [mg/m^3]


def ppm_mg(Pressure, Temperature, Concentration):
    ppm = Concentration/1000000
    Conc_mg_per_m3 = (Pressure/(R*Temperature))*(MW_mg)*ppm
    return Conc_mg_per_m3

# Calculating the Area [m^2]


def Area(D):
    Area = (1/4)*np.pi*(D**2)
    return Area

# Calculating Steady State Flux [g/day*m^2]


def SSFlux_Equ(Q, Cb, Ca, Area):
    Flux = (Q*(Cb-Ca))/Area  # mg/s*m^2
    return Flux


def mg_s_to_g_day_conversion(flux):
    Flux_g_day = flux*84.6
    return Flux_g_day

# Calculating Transient Flux [g/day*m^2]         1 Minute Time Intervals


def Trans_Flux_Equ(Q, Cb, Ca, Slope, Area, Volume):
    Flux = (Q*(Cb-Ca))/Area
    Flux_hr_day = (Flux + (((Slope*Volume))/Area)/60)  # mg/s*m^2
    return Flux_hr_day


def Unc_Flux(Q, U_Q, Cb, U_Cb, Ca, U_Ca, Area):
    U_Flow = (((Cb-Ca)/A)**(2)*(U_Q)**2)
    U_Con_Box = ((Q/A)**(2)*(U_Cb)**2)
    U_Con_Ambient = ((-Q/A)**(2)*(U_Ca)**2)
    U_Flux = np.sqrt(U_Flow + U_Con_Box + U_Con_Ambient)
    return U_Flux


def SEM_calc(SD, N):
    SEM = SD/np.sqrt(N)
    return SEM


def Quotient_Method(x, y):
    Unc = np.sqrt((x)**2 + (y)**2)
    return Unc

# %% Constants for Equations


P = 78900  # Pressure [Pa]
R = 8.3145  # Universal Gas Constant [N*m/mol*K]
MW = 16.04  # Molar weight [g/mol]
MW_mg = MW*1000  # Molar weight [mg/mol]
D_m = 0.4699  # Diameter [m]
A = Area(D_m)  # Area [m^2]
H = 0.111  # Hieght [m]
Vol = Volume(H, A)  # Volume [m^2]

# %% Inputs:Uploading Data,Date, and uncertainties
# Uploading and Cleaning Dataset
read = input("Search for File? (T/F):")
name = input("Report Name:")
date = input("Date:")
if read == "T":
    print("File Window may be hidden by Spyder\n\n minimize spyder to find window")
    root = tk.Tk()
    root.withdraw()
    print("Concentration File (.CSV)")
    Sensors = filedialog.askopenfile(title="Concentration Data", filetypes=(
        ("CSV files", "*.csv"), ("all files", "*.*")))
    Sensors = pd.read_csv(Sensors.name)
else:
    # Transient and Steady State (CSV)(in).csv
    Sensors = pd.read_csv(
        "Box_Concentrations/BOX_Concentration-3-24-field.csv")
# Sensors = pd.read_csv("Transient(Sheet1)2.csv")
# Sensors = pd.read_csv("Steady State (CSV)3.csv")
# Sensors = pd.read_csv("Transient and Steady State (CSV).csv")

Sensors = Sensors.set_index(pd.to_datetime(Sensors["Time"]))
Sensors["Flow Rate"] = Sensors["Flow Rate"].astype(float)*(0.001/60)

# Date of data collection

# Contributing Uncertainties
RMS_Cb = Sensors["B_RMSE"]  # RMS [ppm]
RMS_Ca = Sensors["A_RMSE"]  # RMS [ppm]

Temp_B = Sensors["Temperature"]  # Deg Celcius
Temp_A = Sensors["A_TEMP"]
RMS_Cb_mg_m3 = ppm_mg(Sensors["B_Pressure"],
                      Sensors["A_TEMP"], RMS_Cb)  # RMS [mg/m^3]
RMS_Ca_mg_m3 = ppm_mg(Sensors["A_Pressure"],
                      Sensors["Temperature"], RMS_Ca)  # RMS [mg/m^3]


# %% Time Averaging Data

# Time average Data
interval = "1min"
Sensors = Sensors.resample(interval).last()  # TIME AVERAGING THE SENSOR DATA
Sensors.dropna(axis=0, inplace=True)

# %% Ambient Concentration Graph,Mean,Standard Deviation

# Calculating the Mean and Standard Deviation
Ambient_Avg = np.mean(Sensors["Ambient"])
Ambient_SD = np.std(Sensors["Ambient"])
Ambient_Avg_array = np.ones(len(Sensors['Ambient']))*Ambient_Avg
print(f'\nThe mean ambient concentration is {round(Ambient_Avg,2)}[ppm].')
print(
    f'The standard deviation of the ambient concentration is {round(Ambient_SD,2)}[ppm].')


# %% Box Concentration Graph,Mean,Standard Deviation

# Calculating the Mean and Standard Deviation
Box_Avg = np.mean(Sensors["Box Concentration"])  # ppm
Box_SD = np.std(Sensors["Box Concentration"])  # ppm
Box_Avg_array = np.ones(len(Sensors['Box Concentration']))*Box_Avg
print(f'\nThe mean box concentration is {round(Box_Avg,2)}[ppm].')
print(
    f'The standard deviation of the box concentration is {round(Box_SD,2)}[ppm].')

# %% Converting Concetration Units and Calculating Flux

# Converting Temperatures
Ambient_Temp = CtoK(Sensors["A_TEMP"])
Box_Temp = CtoK(Sensors["Temperature"])

# Converting Concentrations from [ppm] to [mg/m^3]
ca = ppm_mg(Sensors['A_Pressure'], Ambient_Temp, Sensors['Ambient'])
cb = ppm_mg(Sensors['B_Pressure'], Box_Temp, Sensors['Box Concentration'])

Box_Avg_mg_m3 = np.mean(cb)  # mg/m^3
Box_SD_mg_m3 = np.std(cb)
Box_Avg_array_mg_m3 = np.ones(len(Sensors['Box Concentration']))*Box_Avg_mg_m3

Ambient_Avg_mg_m3 = np.mean(ca)  # mg/m^3
Ambient_SD_mg_m3 = np.std(ca)
Ambient_Avg_array_mg_m3 = np.ones(len(Sensors['Ambient']))*Ambient_Avg_mg_m3


# %% Find the Slope Between Each Point in the Box Cencentration Array

# Variables for the Regression
count_array = np.linspace(1, len(Sensors["Box Concentration"]), len(
    Sensors["Box Concentration"]))  # INDEPENDENT
Box_Conc_array = cb  # DEPENDENT
Ambient_Conc_array = ca

# Arrays to store slopes and intercepts for the regressions
slopes = []
intercepts = []

for i in range(len(Box_Conc_array) - 2):  # Running Slope Calculation
    x_pair = np.array([count_array[i], count_array[i+1],
                      count_array[i+2]]).reshape(-1, 1)
    y_pair = np.array(
        [Box_Conc_array[i], Box_Conc_array[i+1], Box_Conc_array[i+2]])

    # Fit the regression model for the pair
    regr = linear_model.LinearRegression()
    regr.fit(x_pair, y_pair)

    # Store slope and intercept
    slopes.append(regr.coef_[0])
    intercepts.append(regr.intercept_)

# Running average
Run_avg = []
Run_avg_Ambient = []

for i in range(len(Box_Conc_array)-2):
    run_avg = np.mean(
        [Box_Conc_array[i], Box_Conc_array[i+1], Box_Conc_array[i+2]])
    run_avg_ambient = np.mean(
        [Ambient_Conc_array[i], Ambient_Conc_array[i+1], Ambient_Conc_array[i+2]])

    # Store averages
    Run_avg.append(run_avg)
    Run_avg_Ambient.append(run_avg_ambient)

# Creating a Transient Calculation Dataframe to track timestamps
Trans_Calc_Dataframe = pd.DataFrame(
    [Sensors["Time"], Sensors["Flow Rate"], cb, ca])
Trans_Calc_Dataframe = Trans_Calc_Dataframe.transpose()
Trans_Calc_Dataframe = Trans_Calc_Dataframe.rename(
    columns={"Unnamed 0": 'Box[mg/m^3]', "Unnamed 1": 'Ambient[mg/m^3]'})
Trans_Calc_Dataframe["Rolling Avg Box"] = Trans_Calc_Dataframe['Box[mg/m^3]'].rolling(
    window=3).mean()
Trans_Calc_Dataframe["Rolling Avg Ambient"] = Trans_Calc_Dataframe['Ambient[mg/m^3]'].rolling(
    window=3).mean()
Final_Transient_Response = Trans_Calc_Dataframe.dropna()


# Convert to arrays
slopes = np.array(slopes)  # mg/m^3*time
intercepts = np.array(intercepts)

# Average the Flux Rates the same way
Avg_Flow = np.mean(Sensors['Flow Rate'])
Flow_Avg_array = np.ones(len(slopes))*Avg_Flow
Cb_array_trans = np.ones(len(slopes))*Run_avg
Ca_array_trans = np.ones(len(slopes))*Run_avg_Ambient


# %% Calculating Transient and Steady State Flux

# Calculating Steady State Flux
flux = SSFlux_Equ(Sensors['Flow Rate'], cb, ca, A)

# Calc Transient Flux Response
Trans = Trans_Flux_Equ(Flow_Avg_array, Cb_array_trans,
                       Ca_array_trans, slopes, A, Vol)

# Unit Conversion mg/s*m^2 to g/day*m^2
Final_SS_Flux = mg_s_to_g_day_conversion(flux)
Final_Transient_Flux = mg_s_to_g_day_conversion(Trans)

# Transient Dataframe with Timestamps for Uncertaintry Calculations
# May not need for methane concetration exported csv file
Final_Transient_Response = Final_Transient_Response.drop(
    ["Rolling Avg Box", "Rolling Avg Ambient"], axis=1)
Final_Transient_Response["Transient Response [g/day*m^2]"] = Final_Transient_Flux

# Averages
SS_avg = np.mean(Final_SS_Flux)
Trans_avg = np.mean(Final_Transient_Flux)

# %% Uncertainty

# SEM for Flow Rate
U_Q_SEM = SEM_calc(np.std(Sensors["Flow Rate"]), len(Sensors["Flow Rate"]))

# Combining SEM and RMS
SEM_Cb = SEM_calc(Box_SD_mg_m3, len(Sensors['Box Concentration']))  # mg/m^3
SEM_Ca = SEM_calc(Ambient_SD_mg_m3, len(Sensors['Ambient']))  # mg/m^3
Unc_Cb = Quotient_Method(SEM_Cb, RMS_Cb_mg_m3)
Unc_Ca = Quotient_Method(SEM_Ca, RMS_Ca_mg_m3)
Unc_Q = Quotient_Method(0.0011, U_Q_SEM)

# Temp and Pressure Def
Temp = CtoK(Sensors["Temperature"])
Press_Box = Sensors['B_Pressure']
Press_Ambient = Sensors['A_Pressure']

# Contributing Uncertainties
U_Q = 0.0011
# U_Cb = ppm_mg(Press_Box[0], Temp[0], Unc_Cb)
# U_Ca = ppm_mg(Press_Box[0], Temp[0], Unc_Ca)

# Creating arrays from Transient Dataframe
Trans_box_con_array = np.array(Final_Transient_Response['Box[mg/m^3]'])
Trans_amb_con_array = np.array(Final_Transient_Response['Ambient[mg/m^3]'])
Trans_flow = np.array(Final_Transient_Response['Flow Rate'])

# Remove first and last from Trans Uncertainty
n = len(Unc_Cb)-1

# Calc Flux Uncertainty
U_Flux = Unc_Flux(Sensors['Flow Rate'], Unc_Q, cb, Unc_Cb, ca, Unc_Ca, A)
U_Flux_trans = Unc_Flux(Trans_flow.astype(int), Unc_Q, Trans_box_con_array.astype(
    int), Unc_Cb[1:n], Trans_amb_con_array.astype(int), Unc_Ca[1:n], Area)

Final_Transient_Response["Transient Uncertainty"] = U_Flux_trans

# %% Final Flux Data Frames
Final_Transient_Response = Final_Transient_Response.drop(
    ["Flow Rate", "Box[mg/m^3]", "Ambient[mg/m^3]"], axis=1)

Final_SS_Response = pd.DataFrame(
    Final_SS_Flux, columns=["SS Box Flux [g/day*m^2]"])
# Final_SS_Response = Final_SS_Flux.rename("SS Box Flux [g/day*m^2]")
Final_SS_Response["Steady State Uncertainty"] = U_Flux

# %% Hourly Averages
# Time average Data
interval_hr = "60min"
# %%  Calc for concentation at min flux
# Time_interval = Sensors.loc["2025-03-24 17:00:00":"2025-03-24 17:59:00"]
# Hourly_ppm_avg = np.mean(Time_interval["Box Concentration"])
# Hourly_Q_avg = np.mean(Time_interval["Flow Rate"])
# Hourly_ambient_ppm_avg = np.mean(Time_interval["Ambient"])
# Hourly_temp_avg = np.mean(Time_interval["A_TEMP"])
# Hourly_temp_internal_avg = np.mean(Time_interval['Temperature'])
# Hourly_ambient_pressure_avg = np.mean(Time_interval["A_Pressure"])
# print(Hourly_ppm_avg)
# print(Hourly_Q_avg)
# print(Hourly_ambient_ppm_avg)
# print(Hourly_temp_avg)
# print(Hourly_temp_internal_avg)
# print(Hourly_ambient_pressure_avg)
# %%
Final_SS_Response_hr = Final_SS_Response.resample(
    interval_hr).last()  # TIME AVERAGING THE SENSOR DATA
Final_SS_Response_hr.dropna(axis=0, inplace=True)

Final_Transient_Response_hr = Final_Transient_Response.resample(
    interval_hr).last()  # TIME AVERAGING THE SENSOR DATA
Final_Transient_Response_hr.dropna(axis=0, inplace=True)

# Average of the Hourly Averages
Hour_Average_Transient = np.mean(
    Final_Transient_Response_hr["Transient Response [g/day*m^2]"])
Hour_Average_SS = np.mean(Final_SS_Response_hr)

# %% Plotting

# Plot for the Average Box Concentration
fig1 = plt.figure(dpi=300)
ax1 = fig1.add_subplot(111)
ax1.plot(Sensors.index, Sensors['Box Concentration'], label="Box Concentation")
ax1.plot(Sensors.index, Box_Avg_array,
         label='Average = {}[ppm]'.format(round(Box_Avg, 2)))
plt.ylabel('CH4 Concentration [ppm]')
plt.xlabel('Time [D H:M]')
fig1.autofmt_xdate()
ax1.set_xlim(Start_Time, End_Time)
plt.legend()
plt.show()
fig1.savefig("Figures/Avg Box Concentation"+date+".png")

# Plot for the Average Ambient Concentration
fig2 = plt.figure(dpi=300)
ax2 = fig2.add_subplot(111)
ax2.plot(Sensors.index, Sensors['Ambient'], label='Ambient Concentration')
ax2.plot(Sensors.index, Ambient_Avg_array,
         label='Average = {}[ppm]'.format(round(Ambient_Avg, 2)))
plt.ylabel('CH4 Concentration [ppm]')
plt.xlabel('Time [D H:M]')
plt.legend()
ax2.set_xlim(Start_Time, End_Time)
fig2.autofmt_xdate()
plt.show()
fig2.savefig("Figures/Avg Ambient Concentation"+date+".png")

# Plot Transient and Steady State Flux
top = Final_Transient_Response["Transient Response [g/day*m^2]"]+U_Flux_trans
bottom = Final_Transient_Response["Transient Response [g/day*m^2]"]-U_Flux_trans
# %%
fig3 = plt.figure(dpi=300)
ax3 = fig3.add_subplot(111)
ax3.plot(Sensors.index, Final_SS_Flux, label="Steady State", color="b")
ax3.fill_between(Sensors.index, Final_SS_Flux+U_Flux,
                 Final_SS_Flux-U_Flux, alpha=0.4, color="steelblue")
ax3.plot(Final_Transient_Response.index,
         Final_Transient_Response["Transient Response [g/day*m^2]"], label="Transient", color='r')
ax3.fill_between(Final_Transient_Response.index,
                 top[1:n], bottom[1:n], alpha=0.4, color="orange")
plt.ylabel('CH4 Flux Rates [g/day*m^2]')
plt.xlabel('Time [D H:M]')
ax3.set_xlim(Start_Time, End_Time)
fig3.autofmt_xdate()
#ax3.axhline(1.48, color='k', linestyle='--', linewidth=1.5, label='Delivered Flux')
plt.legend()

# ax3.set_ylim([0.25,2])
#ax3.set_xlim([pd.to_datetime("2025-03-31 10:52:00"),pd.to_datetime("2025-03-31 12:15:00")])

ax3.set_ylim([-0.01,2])
#ax3.set_xlim([pd.to_datetime("2025-03-31 10:15:00"),pd.to_datetime("2025-03-31 10:53:00")])
plt.show()
fig3.savefig("Figures/Flux Rates"+date+".png")
# %%
# Plot Transient and Steady State Flux with 1hour Time Averaging
fig4 = plt.figure(dpi=300)
ax4 = fig4.add_subplot(111)
ax4.plot(Final_Transient_Response_hr.index,
         Final_Transient_Response_hr["Transient Response [g/day*m^2]"], label="Transient", color="r", marker='d')
ax4.plot(Final_SS_Response_hr.index,
         Final_SS_Response_hr["SS Box Flux [g/day*m^2]"], label="Steady State", color="b", marker='d')
ax4.fill_between(Final_SS_Response_hr.index, Final_SS_Response_hr["SS Box Flux [g/day*m^2]"]+Final_SS_Response_hr['Steady State Uncertainty'],
                 Final_SS_Response_hr["SS Box Flux [g/day*m^2]"]-Final_SS_Response_hr['Steady State Uncertainty'], alpha=0.4, color="steelblue")
ax4.fill_between(Final_Transient_Response_hr.index, Final_Transient_Response_hr["Transient Response [g/day*m^2]"]+Final_Transient_Response_hr['Transient Uncertainty'],
                 Final_Transient_Response_hr["Transient Response [g/day*m^2]"]-Final_Transient_Response_hr['Transient Uncertainty'], alpha=0.4, color="orange")
plt.ylabel('CH4 Flux Rates [g/day*m^2]')
plt.xlabel('Time [D H:M]')
ax4.set_xlim(Start_Time, End_Time)
fig4.autofmt_xdate()
plt.legend()
ax4.set_ylim(-0.1,1.5)
plt.show()
fig4.savefig("Figures/Hourly Flux Rates"+date+".png")

# %% PDF Report

pdf = FPDF()
pdf.add_page()
pdf.set_xy(0, 0)
pdf.set_font('arial', 'B', 14)
pdf.cell(60)
pdf.cell(100, 10, "Flux Quantification Report", 0, 2, 'C')
pdf.set_font('arial', "U", 12)
pdf.cell(100, 10, "Flux Rates [g/day*m^2]", 0, 2, 'C')

image_width = 200  # Adjust width as needed
page_width = pdf.w  # Get total page width
x_centered = (page_width - image_width) / 2  # Calculate centered x position

pdf.image("Figures/Flux Rates"+date+".png",
          x=x_centered, y=None, w=image_width, h=150)
pdf.set_font('arial', "", 12)
pdf.cell(100, 10, "Data from " +
         str(Sensors.index[0])+" to "+str(Sensors.index[len(cb)-1]), 0, 3, 'C')
pdf.cell(
    100, 10, "Average Steady State Flux Rate = %0.2f [g/day*m^2]" % SS_avg, 0, 3, 'C')
pdf.cell(
    100, 10, "Average Transient Flux Rate = %0.2f [g/day*m^2]" % Trans_avg, 0, 3, 'C')
pdf.cell(50, 10, "The uncertainty was calculating using the Partial Differential Method.", 0, 3, 'C')

pdf.add_page()
pdf.set_font('arial', "U", 12)
pdf.cell(200, 10, "Hourly Flux Rates [[g/day*m^2]]", 0, 2, 'C')
pdf.image("Figures/Hourly Flux Rates"+date+".png",
          x=x_centered, y=None, w=image_width, h=150)


pdf.add_page()
pdf.set_font('arial', "U", 12)
pdf.cell(200, 10, "Box Concentration [ppm]", 0, 2, 'C')
pdf.image("Figures/Avg Box Concentation"+date+".png",
          x=x_centered, y=None, w=image_width, h=150)
pdf.set_font('arial', "", 12)
pdf.cell(
    200, 10, "Average Box Concentration= %0.2f [ppm]" % Box_Avg, 0, 1, 'C')
pdf.cell(
    200, 10, "Standard Deviation of the Box Concentration= %0.2f [ppm]" % Box_SD, 0, 1, 'C')

pdf.add_page()
pdf.set_font('arial', "U", 12)
pdf.cell(200, 10, "Ambient Concentration [ppm]", 0, 2, 'C')
pdf.image("Figures/Avg Ambient Concentation"+date+".png",
          x=x_centered, y=None, w=image_width, h=150)
pdf.set_font('arial', "", 12)
pdf.cell(
    200, 10, "Average Ambient Concentration= %0.2f [ppm]" % Ambient_Avg, 0, 3, 'C')
pdf.cell(
    200, 10, "Standard Deviation of the Ambient Concentration= %0.2f [ppm]" % Ambient_SD, 0, 3, 'C')

pdf.output('Reports/'+date+'-FluxReport'+name+'.pdf', 'F')

# %% Exporting Data

Final_Transient_Response.to_csv(
    "Flux Quantification/Transient_Flux"+date+".csv")
Final_SS_Response.to_csv("Flux Quantification/SS_Flux"+date+".csv")
