# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 08:39:11 2025

@author: srhill
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
read = input("Search for File? (T/F):")
#read="F"
if read == "T":
    print("File Window may be hidden by Spyder\n\n minimize spyder to find window")
    root = tk.Tk()
    root.withdraw()
    print("\nSensor file (.csv)")
    Sensor_DATA = filedialog.askopenfile(title = "Select Ambient Sensor Data",filetypes=(("csv files","*.csv"),("all files", "*.*")))
    print("\nLICOR file (.csv)")
    LICOR_DATA = filedialog.askopenfile(title = "Select LICOR Data",filetypes=(("csv files","*.csv"),("all files", "*.*")))
    MOX = pd.read_csv(Sensor_DATA.name)
    LICOR = pd.read_csv(LICOR_DATA.name)
else:
    ######################################################
    # -----------------------Inputs--------------------- #
    #Averaged data with setpoint                         #
    MOX = pd.read_csv("SORACOM_Colocate_Data.csv")
    LICOR = pd.read_csv("LICOR_Colocate_Data.csv")#
    #----------------------------------------------------#
    ######################################################

LICOR["DateTime"] = pd.to_datetime(LICOR['DATE'] + ' ' + LICOR['TIME']) #create DATE TIME
LICOR = LICOR.set_index(["DateTime"])#set datetime as index
LICOR = LICOR.drop(["DATAH","SECONDS","NANOSECONDS","NDX","DIAG","REMARK","DATE","TIME","H2O","CO2","RESIDUAL","RING_DOWN_TIME","THERMAL_ENCLOSURE_T","PHASE_ERROR","LASER_T_SHIFT","INPUT_VOLTAGE","CHK","CAVITY_P","CAVITY_T","LASER_PHASE_P","LASER_T"],axis =1)
LICOR.dropna(axis=0, inplace=True)
LICOR = LICOR.resample('1min').mean() #minute average data
MOX = MOX.set_index(pd.to_datetime(MOX["__time"]))
MOX = MOX.drop(["__resourceId","__resourceType","__iso8601Time","aFlow","aTemp","__time"],axis =1)
MOX.dropna(axis=0, inplace=True)
MOX = MOX.resample("1min").last()
#MOX = MOX.loc["2024-08-08 12:00:00":"2024-08-20 12:00:00"]
DATA = MOX
DATA["CH4"] = LICOR["CH4"]/1000
DATA.drop(["B_Gas","B_TEMP","B_Humidity","B_Pressure","B_TGS00","B_TGS02","B_TGS11"],axis = 1,inplace = True)
DATA = DATA.resample('1min').mean()
DATA.dropna(axis=0,inplace = True)
DATA["R1"] = DATA["A_TGS11"]/DATA["A_TGS00"]
DATA["R2"] = DATA["A_TEMP"]/DATA["A_Humidity"]
DATA["R3"] = DATA["A_TGS02"]/DATA["A_TGS11"]
DATA["R4"] = DATA["A_TEMP"]/DATA["A_TGS02"]


#%%
X = DATA[['A_Gas',"A_Humidity",'A_Pressure','A_TEMP',"A_TGS00",'A_TGS02',"A_TGS11","R1","R2","R3","R4"]]
y = DATA['CH4']
reg = linear_model.LinearRegression()
reg.fit(X,y)


reg_model_diff = pd.DataFrame({'Actual value': y, 'Predicted value': reg.predict(X) })

RMS = np.sqrt(mean_squared_error(reg_model_diff["Actual value"],reg_model_diff["Predicted value"]))
print("RMS error is %0.4f [ppm]"%(RMS))
#%% plot
fig1,ax1 = plt.subplots(dpi=400)
ax1.plot(reg_model_diff.index,reg_model_diff["Predicted value"],label = "Sensor Response")
ax1.fill_between(reg_model_diff.index,reg_model_diff["Predicted value"]-RMS,reg_model_diff["Predicted value"]+RMS,alpha=0.25)
ax1.plot(reg_model_diff.index,reg_model_diff["Actual value"],label = "Reference Instrument Response")
ax1.legend()
ax1.set_xlabel("Time")
ax1.set_ylabel("Methane Concentration [ppm]")
fig1.autofmt_xdate()

fig2,ax2 = plt.subplots(dpi=400)
ax2.scatter(reg_model_diff["Actual value"],reg_model_diff["Predicted value"],color = "blue", label = "Sensor Response")
ax2.plot(np.linspace(min(reg_model_diff["Actual value"]),max(reg_model_diff["Actual value"]),100),np.linspace(min(reg_model_diff["Actual value"]),max(reg_model_diff["Actual value"]),100),color = "black",linestyle = "dotted",label = "One to One Line")
ax2.legend()
ax2.set_xlabel("LICOR Concentration [ppm]")
ax2.set_ylabel("Sensor Concentration [ppm]")

#savefigs
fig1.savefig("Figures/Ambient_Reg.png")
fig2.savefig("Figures/Ambient_Reg_121.png")
#%% export coefficients
with open('coefficients/Ambient_Colocate_reg.pkl',"wb") as f:
    pickle.dump(reg,f)
with open("coefficients/A_RMSE.txt","w") as file:
    file.write(str(RMS))
EQ = "Colocation EQ = %0.2f + %0.2f*B_Gas + %0.2f*B_Humidity\n + %0.2f*B_TEMP + %0.2f*B_TGS00 + %0.2f*B_TGS02 + %0.2f*TGS11 + %0.2f*R1 + %0.2f*R2 + %0.2f*R3 + %0.2f*R4"%(reg.intercept_,reg.coef_[0],reg.coef_[1],reg.coef_[2],reg.coef_[3],reg.coef_[4],reg.coef_[5],reg.coef_[6],reg.coef_[7],reg.coef_[8],reg.coef_[9])
#%% PDF Report
pdf = FPDF()
pdf.add_page()
pdf.set_xy(0, 0)
pdf.set_font('arial', 'B', 14)
pdf.cell(60)
pdf.cell(100, 10, "Ambient Colocation Report", 0, 2, 'C')
pdf.set_font('arial',"U", 12)
pdf.cell(100,10,"One to One Plot Of Colocation",0,2,'C')
pdf.image('Figures/Ambient_Reg_121.png', x = 30, y = None, w = 170, h = 125, type = '', link = '')
pdf.set_font('arial',"", 12)
pdf.cell(100,10,"RMSE or Error for Regression is %0.4f [ppm]"%(RMS),0,3,'L')
pdf.set_font('arial',"U", 12)
pdf.cell(100,10,"Colocated Data versus LICOR Signal",0,3,'C')
pdf.image('Figures/Ambient_Reg.png', x = 40, y = None, w = 170, h = 100, type = '', link = '')
pdf.cell(100,30,"Colocation Equation",0,3,'C')
pdf.set_font('arial',"", 7)
pdf.cell(100,0,EQ,0,3,'C')


pdf.output('Reports/Colocation_Report.pdf', 'F')
print("\nReport Generated\n")