# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:40:53 2025
Methane Group Calibration Equation
@author:Sam Hill

----------PLEASE READ----------

DESCRIPTION
This code takes in a spreadsheet of time averaged data for each calibrated 
setpoint and generates a calibration regression to map the sensors responce methane.

INPUTS
    Averaged setpoint data generated using script Get_Setpoints.py, found in /Averaged_Setpoints directory

OUTPUTS
    Plot of regression data responce in methane along with delivered setpoints 
    
    Plot of regression data plotted against delivered methane concentration, aka 1 to 1 plot
    
    3 regression models, reglow.plk, regmid.plk, reghigh.plk in /Coeffiecients
"""


import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle
from pylab import title, figure, xlabel, ylabel, xticks, bar, legend, axis, savefig
from fpdf import FPDF
import tkinter as tk
from tkinter import filedialog

print("File Window may be hidden by Spyder\n\n minimize spyder to find window")
root = tk.Tk()
root.withdraw()
root.lift()
root.attributes('-topmost', True)
print("Averaged Setpoints File (.XLSX)")
Setpoints_file = filedialog.askopenfile(title = "Averaged Setpoints",filetypes=(("XLSX files","*.xlsx"),("all files", "*.*")))
DATA = pd.read_excel(Setpoints_file.name)



#%%


# Data Wrangling, spit into 3 ranges
#training ranges
LOW = DATA[DATA["Setpoint"]<=71]
MED = DATA[DATA["Setpoint"]<=1000]
MED = MED[MED["Setpoint"]>=60]
HIGH = DATA[DATA["Setpoint"]>=600]
#Acutal delevering ranges
#Lo = DATA[DATA["Setpoint"]<=65]
#Mid = DATA[DATA["Setpoint"]<=774]
#Mid = Mid[Mid["Setpoint"]>=65]
#Hi = DATA[DATA["Setpoint"]>=774]

#adjustd to predict for all setpoints in calibration
Lo = DATA[DATA["Setpoint"]<=71]
Mid = DATA[DATA["Setpoint"]<=750]
Mid = Mid[Mid["Setpoint"]>=71]
Hi = DATA[DATA["Setpoint"]>=750]

#no Ratios
#Low Regression
Xlow = LOW[["BO Humidity (%)",'BO Pressure (hPa)','BO Temperature (C)',"BO TGS2600 (mV)","BO TGS2611 (mV)","BO TGS2602 (mV)","BO EC_Worker (mV)"]]
ylow = LOW['Setpoint']
low_reg = linear_model.LinearRegression()
low_reg.fit(Xlow,ylow)

Xlo = Lo[["BO Humidity (%)",'BO Pressure (hPa)','BO Temperature (C)',"BO TGS2600 (mV)","BO TGS2611 (mV)","BO TGS2602 (mV)","BO EC_Worker (mV)"]]
ylo = Lo["Setpoint"]
reg_model_diff_low = pd.DataFrame({'Actual value': ylo, 'Predicted value': low_reg.predict(Xlo) })


#Medium Regression
Xmed = MED[["BO Humidity (%)",'BO Pressure (hPa)','BO Temperature (C)',"BO TGS2600 (mV)","BO TGS2611 (mV)","BO TGS2602 (mV)","BO EC_Worker (mV)","SGX_Digital (ppm)"]]
ymed = MED['Setpoint']
med_reg = linear_model.LinearRegression()
med_reg.fit(Xmed,ymed)

Xmid = Mid[["BO Humidity (%)",'BO Pressure (hPa)','BO Temperature (C)',"BO TGS2600 (mV)","BO TGS2611 (mV)","BO TGS2602 (mV)","BO EC_Worker (mV)","SGX_Digital (ppm)"]]
ymid = Mid["Setpoint"]
reg_model_diff_med = pd.DataFrame({'Actual value': ymid, 'Predicted value': med_reg.predict(Xmid) })

#High Regression
Xhigh = HIGH[["BO Humidity (%)",'BO Pressure (hPa)','BO Temperature (C)',"SGX_Digital (ppm)"]]
yhigh = HIGH['Setpoint']
high_reg = linear_model.LinearRegression()
high_reg.fit(Xhigh,yhigh)

Xhi = Hi[["BO Humidity (%)",'BO Pressure (hPa)','BO Temperature (C)',"SGX_Digital (ppm)"]]
yhi = Hi["Setpoint"]
reg_model_diff_high = pd.DataFrame({'Actual value': yhi, 'Predicted value': high_reg.predict(Xhi)})

#combine stuff
Combined = pd.concat([reg_model_diff_low,reg_model_diff_med,reg_model_diff_high])
Combined = (Combined.reset_index()
        .drop_duplicates(subset='index', keep='last')
        .set_index('index').sort_index())
# calculate RMS for entire regression
RMS = np.sqrt(mean_squared_error(Combined["Actual value"],Combined["Predicted value"]))
print("Overall RMS error is %0.4f [ppm]"%(RMS))
RMSlow = np.sqrt(mean_squared_error(reg_model_diff_low["Actual value"],reg_model_diff_low["Predicted value"]))
RMSmid = np.sqrt(mean_squared_error(reg_model_diff_med["Actual value"],reg_model_diff_med["Predicted value"]))
RMShigh = np.sqrt(mean_squared_error(reg_model_diff_high["Actual value"],reg_model_diff_high["Predicted value"]))
print("Low Reg RMS error is %0.4f [ppm]"%(RMSlow))
print("Mid Reg RMS error is %0.4f [ppm]"%(RMSmid))
print("High Reg RMS error is %0.4f [ppm]"%(RMShigh))
setpointerror = 0.0199
#%%
#plot stuff
fig1,ax1 = plt.subplots(dpi = 300)
ax1.plot(reg_model_diff_low.index,reg_model_diff_low["Predicted value"],label = "Low Range",color = "green")
ax1.fill_between(reg_model_diff_low.index,reg_model_diff_low["Predicted value"]-RMSlow,reg_model_diff_low["Predicted value"]+RMSlow,color = 'green',alpha = 0.25)
ax1.plot(reg_model_diff_med.index,reg_model_diff_med["Predicted value"],label = "Med Range",color = "purple")
ax1.fill_between(reg_model_diff_med.index,reg_model_diff_med["Predicted value"]-RMSmid,reg_model_diff_med["Predicted value"]+RMSmid,color = 'purple',alpha = 0.25)
ax1.plot(reg_model_diff_high.index,reg_model_diff_high["Predicted value"],label = "High Range",color = "orange")
ax1.fill_between(reg_model_diff_high.index,reg_model_diff_high["Predicted value"]-RMShigh,reg_model_diff_high["Predicted value"]+RMShigh,color = 'orange',alpha = 0.25)

ax1.plot(DATA.index,DATA["Setpoint"],color = "black",label = "Setpoints",linestyle = "dotted", alpha = 0.7)
ax1.fill_between(DATA.index,DATA["Setpoint"]+setpointerror,DATA["Setpoint"]-setpointerror,alpha=0.25,color = "black")
ax1.set_ylabel("Concentration [ppm]")
ax1.legend()
ax1.set_xlabel("Setpoint Number")
ax1.set_ylabel("Concentration Methane [ppm]")
#ax1.set_ylim([0,700])

Combined["humidities"] = DATA["BO Humidity (%)"]
Combined["temperature"] = DATA['BO Temperature (C)']
Combined["color"] = "black"
for i in range(len(Combined)):
    if Combined.iloc[i]["temperature"] < 10:
        Combined.loc[i,"color"] = "midnightblue"
    if Combined.iloc[i]["temperature"] > 20 and Combined.iloc[i]["temperature"] < 30:
        Combined.loc[i,"color"] = "blue"
    if Combined.iloc[i]["temperature"] > 30 and Combined.iloc[i]["temperature"] < 36:
        Combined.loc[i,"color"] = "cornflowerblue"
    if Combined.loc[i,"temperature"] > 36:
        Combined.loc[i,"color"]="lightblue"

#%%
#121 plot
fig2,ax2 = plt.subplots(dpi=300)
ax2.plot(np.linspace(0,10000,100),np.linspace(0,10000,100),label = "1 to 1",alpha = 0.75, linestyle = "--",color = "black")
ax2.scatter([0],[0],color = "midnightblue",label = "5 [C]")
ax2.scatter([0],[0],color = "blue",label = "25 [C]")
ax2.scatter([0],[0],color = "cornflowerblue",label = "35 [C]")

ax2.fill_between(reg_model_diff_low["Actual value"],reg_model_diff_low["Predicted value"]-RMSlow,reg_model_diff_low["Predicted value"]+RMSlow,color = "orange",alpha = 0.25)

ax2.fill_between(reg_model_diff_med["Actual value"],reg_model_diff_med["Predicted value"]-RMSmid,reg_model_diff_med["Predicted value"]+RMSmid,color = "orange",alpha = 0.25)


ax2.fill_between(reg_model_diff_high["Actual value"],reg_model_diff_high["Predicted value"]-RMShigh,reg_model_diff_high["Predicted value"]+RMShigh,color = "orange",alpha = 0.25)
ax2.scatter(Combined["Actual value"],Combined["Predicted value"],c = Combined['color'])

ax2.legend()
ax2.set_xlabel("Actual Value [ppm]")
ax2.set_ylabel("Predicted Value [ppm]")
ax2.set_ylim([0,1000])
ax2.set_xlim([0,1000])
#%%
fig3,ax3 = plt.subplots(dpi=300)
ax3.loglog(np.linspace(0,10000,100),np.linspace(0,10000,100),color = "black",linestyle="dashed")
ax3.loglog(reg_model_diff_low["Actual value"],reg_model_diff_low["Predicted value"],color="green")
ax3.fill_between(reg_model_diff_low["Actual value"],reg_model_diff_low["Predicted value"]-RMSlow,reg_model_diff_low["Predicted value"]+RMSlow,color = "green",alpha = 0.25)
ax3.loglog(reg_model_diff_med["Actual value"],reg_model_diff_med["Predicted value"],color = "blue")
ax3.fill_between(reg_model_diff_med["Actual value"],reg_model_diff_med["Predicted value"]-RMSmid,reg_model_diff_med["Predicted value"]+RMSmid,color = "blue",alpha = 0.25)
ax3.loglog(reg_model_diff_high["Actual value"],reg_model_diff_high["Predicted value"],color = "orange")
ax3.fill_between(reg_model_diff_high["Actual value"],reg_model_diff_high["Predicted value"]-RMShigh,reg_model_diff_high["Predicted value"]+RMShigh,color = "orange",alpha = 0.25)

#Equations
#lowEQ = "Low EQ = %0.2f + %0.2f*B_Gas + %0.2f*B_Humidity\n + %0.2f*B_TEMP + %0.2f*TGS00 + %0.2f*B_TGS02 + %0.2f*TGS11 "%(low_reg.intercept_,low_reg.coef_[0],low_reg.coef_[1],low_reg.coef_[2],low_reg.coef_[3],low_reg.coef_[4],low_reg.coef_[5])
#midEQ = "Middle EQ = %0.2f + %0.2f*B_Gas + %0.2f*B_Humidity\n + %0.2f*B_TEMP + %0.2f*TGS00 + %0.2f*B_TGS02 + %0.2f*TGS11 "%(med_reg.intercept_,med_reg.coef_[0],med_reg.coef_[1],med_reg.coef_[2],med_reg.coef_[3],med_reg.coef_[4],med_reg.coef_[5])
#HighEQ = "High EQ = %0.2f +  %0.2f*B_Humidity\n + %0.2f*B_TEMP + %0.2f*SGX_CH4 "%(high_reg.intercept_,high_reg.coef_[0],high_reg.coef_[1],high_reg.coef_[2])

#%% save shit
fig1.savefig("Figures/Linear_Regression.png")
fig2.savefig("Figures/Linear_Regression_1to1.png")

with open('coefficients/lowreg_Cal3.pkl',"wb") as f:
    pickle.dump(low_reg,f)
with open('coefficients/midreg_Cal3.pkl',"wb") as f:
    pickle.dump(med_reg,f)
with open('coefficients/highreg_Cal3.pkl',"wb") as f:
    pickle.dump(high_reg,f)
with open("coefficients/B_RMSE_L_Cal3.txt","w") as file:
    file.write(str(RMSlow))
with open("coefficients/B_RMSE_M_Cal3.txt","w") as file:
    file.write(str(RMSmid))
with open("coefficients/B_RMSE_H_Cal3.txt","w") as file:
    file.write(str(RMShigh))    
    
#%% PDF Report
pdf = FPDF()
pdf.add_page()
pdf.set_xy(0, 0)
pdf.set_font('arial', 'B', 14)
pdf.cell(60)
pdf.cell(100, 10, "Calibration Report", 0, 2, 'C')
pdf.set_font('arial',"U", 12)
pdf.cell(100,10,"One to One Plot Of Calibration",0,2,'C')
pdf.image('Figures/Linear_Regression_1to1.png', x = 30, y = None, w = 150, h = 125, type = '', link = '')
pdf.set_font('arial',"", 12)
pdf.cell(0,10,"RMSE or Error for low Regression is %0.4f [ppm]"%(RMSlow),0,3,'L')
pdf.cell(0,10,"RMSE or Error for mid Regression is %0.4f [ppm]"%(RMSmid),0,3,'L')
pdf.cell(0,10,"RMSE or Error for high Regression is %0.4f [ppm]"%(RMShigh),0,3,'L')

pdf.set_font('arial',"", 12)
#pdf.cell(100,100,"Low, Medium, and High Regression Responce to Setpoints",0,3,'C')
pdf.image('Figures/Linear_Regression.png', x = 40, y = None, w = 125, h = 100, type = '', link = '')
pdf.set_font('arial',"U", 12)
pdf.cell(100,30,"Calibration Equations",0,3,'C')
pdf.set_font('arial',"", 8)
#pdf.cell(100,10,lowEQ,0,3,'C')
#pdf.cell(90,10,midEQ,0,3,'C')
#pdf.cell(100,10,HighEQ,0,3,'C')

pdf.output('Reports/Cal1-Calibration_Report.pdf', 'F')
print("\nReport Generated\n")
#%% uncomment to generate residual file
Combined["humidities"] = DATA["BO Humidity (%)"]
Combined["temperature"] = DATA["BO Temperature (C)"]
#Combined["pressure"] = DATA["B_Pressure"]
Combined["H2S"] = DATA["H2S"]
Combined.to_csv("Residuals.csv")