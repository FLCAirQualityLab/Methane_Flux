# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:51:41 2025
Methane Quantification Code 
@author: Sam Hill

----------PLEASE READ----------

DESCRIPTION
This script takes in Flux Chamber Data and then combines them and applies calibration to data. 
It then outputs the raw data as BOX_Concentration+Date.xlsx along with a Report with a Figure,
 RMS value and Maximum and Minimum Concentrations
 
Applies calibration equations for both internal and external chambers
 
 INPUT: 
     Regression Models: lowreg.plk , midreg.plk, highreg.plk, Ambient_Colocate_reg.plk all saved in the directory /coefficients
     
     RMSE error: B_RMSE.txt and A_RMSE.txt
     
     Data: LICOR as .csv
     
     User Inputs: Start Time, End Time, Sheet Name, default name is Data.
 
 OUTPUT:
     Methane Responce saved in /Box_Concentrations as Box_Concentration-"date".xlsx
     
     Plot of methane Responce saved in /Figures as MethaneConc.png
     
     Report of Methane Quantification saved in /Reports as MethaneReport
     
"""

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import linear_model
import tkinter as tk
from tkinter import filedialog
import pickle
from pylab import title, figure, xlabel, ylabel, xticks, bar, legend, axis, savefig
from fpdf import FPDF
import copy as cp
import numpy as np
import gspread

#%% INPUTS
################# INPUTS ############################
Start_Time = pd.to_datetime("2025-04-24 8:50:00") #start time
End_Time = pd.to_datetime("2025-04-24 11:45:00") #end time
sheet = "Field Deployment 4-23_4-24"  #sheet name
######################################################
#%%

credentials = {
    "type": "service_account",
  "project_id": "particle-integration-450619",
  "private_key_id": "c33895277c5a8e902cc008c6d8ae9c4ca0a832fb",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDLYC6AdsfLQMxZ\nfUBWVM+GzRHe86XjMvfrueuUwWa9rlKB975thOjK55ted9qva2b4UU61yyd4WfMU\nQvKGNGnL7zi2/Jl3rucnN5z9m6yWuUWkf+5BE6HHEXC4TSkOHaw1ZJcduUXJim8v\n7uazCOi6dFaPF6+4m0EsJHH1OuC2hiZGQzMe2uoWVd+WI3GBoPVTK6pIwxIt5+Qg\ns05+RW6Sfu7l2VPHXa0TvIXhMsdxaPI/NTT2i84m6TEblKkS6ehU9dpcqbiWQLmy\ngfxmUA/lZ4W4zBMp4ZMXnw0CsjIVJ9Mj8jjIKkIMna36+72x0Mh2SsAe6MHnZEly\nDt4SALA1AgMBAAECggEAAwiHEs5EgW3fEfEtjIFrRGn2+6HGZkOTbf4MiWAWhh6P\nEfnmxw+PjzBK+lyWGu+/A2QuZO/n5Te4O3H+EL+HUsNELEHZgj0qjn19fnefhORQ\nsKAbEXg+hIvvdibqBW2SQjbUyA6Sf89IrxcWSfKWtVwcPLoSBPXTnlLYRw2LVzA4\n1DY2c+PZ3czZg87Go+roHTcRjYztNQk5ERFsmBPZ2t7VVxDZ+gF5BrZazl/BtsrJ\nFeLQLs+tmBS566KserCqwjAKL35Mot8A3aJgRuMd0KLhwtdwQzf+0CRa5him0jyP\nug/20QJotQ7AIOQkS6PkQw71jpFjEL644eZjDHJEUQKBgQDr5ft0SDDVCqdCUU9n\nXikr7JPN6PJuUci0e3wpyOip6Twx2iHonFYE6uDpQYXc6udrBrBKD2PsmVmGcNP2\nnZjpXwOujkJv2wrE7xZCr34AWgBaw1k5tDHJpvXSEfJYE3yxaPkKcw1hNYdQrPJq\nNpOuHGbsBX/bzUqKtauTT5NSEQKBgQDctLtrJixgM0+FMsSwO/ymJowT8EtrEt61\nwa13FxDDGsJ0Q5MerCcRuOmIBOtzZid3djzczO3wTAnmnsDe6kKVQ5CaWTbkq4D8\nk7iDtzKa8Kudtt4ivkZIpNRnaSMxBeHg4ozfmER6QcHYf7viwRDpPk+Kg0aViHtI\ntBFFyIzX5QKBgGbZ6R6z8waQFIjnprUs6sqJ5Y+bh0fuRJHcXTBitn6OgH5D2xDu\nSNrwBYvt042Upb2WNvqzZx/bZJsnSmN2JxtpH9PVlsXqPPHMRGpi1Y8Vrp3kGlz8\nYdDICNnElWMta+p2GE8kBqthiVP1c+Q3U4BQRdeWNj8BbQS5XMtnoJXhAoGAdvEz\nu/gTnDiq11bX5z4813IYtbofevHtcjiReofEsdDfEPdF9xbB62wi9vnnxgY8qMXg\n4QHUDVd9Unsl6DyJa5XA/V2tFqlS10vo+ZsmO5gOdO2TY1f12rpx+dUQcSABbkfJ\nscqGxPhhNoMIR5jSu1CoXzaGOJoYDsN0N45wUP0CgYEAz2F0AXcK5+ZymRfBYVJa\nUJ8+2j0Q9nmiV79Jd0F0klTOVdahCf09dxblHWLl8vFU++P9mPV5hNxtSiLewbLj\nQN5XJinJItK+uIV3AxUaAsBMwVlsyTOoaimC5N7zCAhXGfpzAEvIVAeorWFSxP2p\nWNFUeagpQr+e3qoBJsX7JiI=\n-----END PRIVATE KEY-----\n",
  "client_email": "boroniuspull@particle-integration-450619.iam.gserviceaccount.com",
  "client_id": "104286311815340018688",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/boroniuspull%40particle-integration-450619.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
gc = gspread.service_account_from_dict(credentials)

sh = gc.open("BoroniusData")
def sheet_dataframe(sheet):
    worksheet = sh.worksheet(sheet)
    dataframe = pd.DataFrame(worksheet.get_all_records())
    return dataframe
MOX = sheet_dataframe(sheet)




#uncomment for user inputs


date = input("todays Date (M-D):")
#date="test-test"

#%%
Lic = input("Compare Licor Data? (T/F):")
if Lic == "T":
    print("LICOR DATA (.CSV)")
    LICOR = filedialog.askopenfile(title = "LICOR",filetypes=(("CSV files","*.csv"),("all files", "*.*")))
    LICOR = pd.read_csv(LICOR.name)
    LICOR["DateTime"] = pd.to_datetime(LICOR['DATE'] + ' ' + LICOR['TIME']) #create DATE TIME
    LICOR = LICOR.set_index(["DateTime"])#set datetime as index
    LICOR = LICOR.drop(["DATAH","SECONDS","NANOSECONDS","NDX","DIAG","REMARK","DATE","TIME","H2O","CO2","RESIDUAL","RING_DOWN_TIME","THERMAL_ENCLOSURE_T","PHASE_ERROR","LASER_T_SHIFT","INPUT_VOLTAGE","CHK","CAVITY_P","CAVITY_T","LASER_PHASE_P","LASER_T"],axis =1)
    LICOR.dropna(axis=0, inplace=True)



#MOX["B_CH4"] = 0

MOX = MOX.set_index(pd.to_datetime(MOX["Time"])) #__time
MOX = MOX.drop(["Time"],axis =1)
#MOX.loc[Start_Time:End_Time]
#SGX = SGX.set_index(SGX["Date & Time"])
#SGX.drop(["Date & Time","Fault Code"], axis = 1,inplace = True)

# Time average dATA
for i in range(len(MOX)):
    if MOX["SGX_Digital (ppm)"][i] == 404:
        MOX["SGX_Digital (ppm)"][i] = np.nan
MOX.dropna(axis=0,inplace=True)

interval = "60s"
MOX = MOX.resample(interval).last()
#SGX = SGX.resample(interval).last()
MOX.dropna(axis=0, inplace=True)
#SGX.dropna(axis=0,inplace=True)
DATA = MOX
#DATA["CH4"] = SGX["Concentration"]
#DATA.dropna(axis=0,inplace=True)

with open('coefficients/lowreg_Cal3.pkl', 'rb') as f:
      reg_low = pickle.load(f)
with open('coefficients/midreg_Cal3.pkl', 'rb') as f:
      reg_mid = pickle.load(f)
with open('coefficients/highreg_Cal3.pkl', 'rb') as f:
      reg_high = pickle.load(f)
with open("coefficients/B_RMSE_L_Cal3.txt","r") as f:
      RMSlow = float(f.read())
with open("coefficients/B_RMSE_M_Cal3.txt","r") as f:
      RMSmid = float(f.read())
with open("coefficients/B_RMSE_H_Cal3.txt","r") as f:
      RMShigh = float(f.read())
TS = DATA.index
#%% Apply Ambient Regression
with open('coefficients/Ambient_Colocate_reg.pkl', 'rb') as f:
      reg_amb = pickle.load(f)
with open("coefficients/A_RMSE.txt","r") as f:
      A_RMS = float(f.read())
DATA["R1"] = DATA["Main TGS2611 (mV)"]/DATA["Main TGS2600 (mV)"]
DATA["R2"] = DATA["Temperature (C)"]/DATA["Humidity (%)"]
DATA["R3"] = DATA["Main TGS2602 (mV)"]/DATA["GasResistance (Ohms)"]
DATA["R4"] = DATA["Temperature (C)"]/DATA["Main TGS2602 (mV)"]

DATA["A_Gas"] = DATA["GasResistance (Ohms)"]
DATA["A_Humidity"] = DATA["Humidity (%)"]
DATA["A_Pressure"] = DATA["Pressure (hPa)"]
DATA["A_TEMP"] = DATA["Temperature (C)"]
DATA["A_TGS00"] = DATA["Main TGS2600 (mV)"]
DATA["A_TGS02"] = DATA["Main TGS2602 (mV)"]
DATA["A_TGS11"] = DATA["Main TGS2611 (mV)"]
A_COEF = DATA[['A_Gas',"A_Humidity",'A_Pressure','A_TEMP',"A_TGS00",'A_TGS02',"A_TGS11","R1","R2","R3","R4"]]
A_CH4 = pd.DataFrame({"A_CH4":reg_amb.predict(A_COEF)}).set_index(TS)

#%% Apply calibration regressions




COEF_L = DATA[["BO Humidity (%)",'BO Pressure (hPa)','BO Temperature (C)',"BO TGS2600 (mV)"]]
COEF_L["BO TGS2611 (mV)"] = DATA["BO TGS2602 (mV)"]
COEF_L["BO TGS2602 (mV)"] = DATA["BO TGS2611 (mV)"]
COEF_L["BO EC_Worker (mV)"]=DATA["BO EC_Worker (mV)"]

COEF_M = DATA[["BO Humidity (%)",'BO Pressure (hPa)','BO Temperature (C)',"BO TGS2600 (mV)"]]
COEF_M["BO TGS2611 (mV)"] = DATA["BO TGS2602 (mV)"]
COEF_M["BO TGS2602 (mV)"] = DATA["BO TGS2611 (mV)"]
COEF_M["BO EC_Worker (mV)"]=DATA["BO EC_Worker (mV)"]
COEF_M["SGX_Digital (ppm)"] = DATA["SGX_Digital (ppm)"]
#%%
#COEF_L = DATA[["BO Humidity (%)",'BO Pressure (hPa)','BO Temperature (C)',"BO TGS2600 (mV)","BO TGS2602 (mV)","BO TGS2611 (mV)","BO EC_Worker (mV)"]]
#COEF_M = DATA[["BO Humidity (%)",'BO Pressure (hPa)','BO Temperature (C)',"BO TGS2600 (mV)","BO TGS2611 (mV)","BO TGS2602 (mV)","BO EC_Worker (mV)","SGX_Digital (ppm)"]]

COEF_H = DATA[["BO Humidity (%)",'BO Pressure (hPa)','BO Temperature (C)',"SGX_Digital (ppm)"]]
high = pd.DataFrame({"CH4":reg_high.predict(COEF_H)}).set_index(TS)
mid = pd.DataFrame({"CH4":reg_mid.predict(COEF_M)}).set_index(TS)
low = pd.DataFrame({"CH4":reg_low.predict(COEF_L)}).set_index(TS)
high["RMSE"]=RMShigh
mid["RMSE"]=RMSmid
low["RMSE"]=RMSlow
#adjust to zero each range
# low["CH4"] = low["CH4"]+310
# mid["CH4"]=mid["CH4"]-1325
methane = cp.deepcopy(mid)*0




for i in range(len(methane)):
    
    
    if low.iloc[i]["CH4"] < 75 and low.iloc[i]["CH4"] >= 2 :
        methane.iloc[i]["CH4"] = low.iloc[i]["CH4"]
        methane.iloc[i]["RMSE"] = low.iloc[i]["RMSE"]
    elif mid.iloc[i]["CH4"] >= 75 and mid.iloc[i]["CH4"] < 700:
        methane.iloc[i]["CH4"] = mid.iloc[i]["CH4"]
        methane.iloc[i]["RMSE"] = mid.iloc[i]["RMSE"]
    elif high.iloc[i]["CH4"] >= 700 and high.iloc[i]["CH4"] <= 11000:
        methane.iloc[i]["CH4"] = high.iloc[i]["CH4"]
        methane.iloc[i]["RMSE"] = high.iloc[i]["RMSE"]
    else:
        methane.iloc[i]["CH4"] = np.nan
        methane.iloc[i]["RMSE"] = np.nan
methane.dropna(axis=0,inplace=True)
#plot stuff
#%%
fig1,ax1 = plt.subplots(dpi=300)
ax1.plot(methane.index,methane["CH4"],color= "orange",label = "Sensor Array")
ax1.fill_between(methane.index,methane["CH4"]+methane["RMSE"],methane["CH4"]-methane["RMSE"],alpha=0.4,color = "orange")
if Lic == "T":
    ax1.plot(LICOR.index,LICOR["CH4"]/1000,label = "Reference Instrument",color="red")
ax1.legend()
ax1.set_xlabel("Time [DD HH:MM]")
ax1.set_ylabel("Methane [ppm]")
fig1.autofmt_xdate()
#ax1.set_ylim(-2,200)
ax1.set_xlim(Start_Time,End_Time)
fig1.savefig("Figures/MethaneConc"+date+".png")

fig2,ax2 = plt.subplots(dpi=300)
ax2.plot(low.index,low["CH4"])
ax2.set_title("Low Reg CH4")
fig2.autofmt_xdate()
#ax2.set_ylim(0,100)
ax2.set_xlim(Start_Time,End_Time)

fig3,ax3 = plt.subplots(dpi=300)
ax3.plot(mid.index,mid["CH4"])
ax3.set_title("Mid Reg CH4")
fig3.autofmt_xdate()
#ax3.set_ylim(75,1000)
ax3.set_xlim(Start_Time,End_Time)

fig4,ax4 = plt.subplots(dpi=300)
ax4.plot(high.index,high["CH4"])
ax4.set_title("High Reg CH4")
fig4.autofmt_xdate()
#ax4.set_ylim(500,11000)
ax4.set_xlim(Start_Time,End_Time)

fig5,ax5 = plt.subplots(dpi=300)
ax5.plot(DATA.index,DATA["BO TGS2600 (mV)"],label = "TGS00")
ax5.plot(DATA.index,DATA["BO TGS2602 (mV)"],label = "TGS02")
ax5.plot(DATA.index,DATA["BO TGS2611 (mV)"],label = "TGS11")
ax5.set_title("MOX sensors")
ax5.legend()
fig5.autofmt_xdate()
ax5.set_xlim(Start_Time,End_Time)


#%% 121 Graph
# interval = "1min"
# LICOR_2 = LICOR.resample(interval).last()
# LICOR_2 = LICOR_2.drop(LICOR_2.loc["2025-03-28 13:53:00":"2025-03-28 13:55:00"].index)
# LICOR_2 = LICOR_2.drop("2025-03-28 15:08:00")
# Methane_2 = methane.iloc[:-1]

# fig6,ax6 = plt.subplots(dpi=300)
# ax6.scatter(methane["CH4"],LICOR_2["CH4"]/1000,label = "TGS00")
# ax6.plot(np.linspace(0,300,100),np.linspace(0,300,100),label = "1 to 1",alpha = 0.75, linestyle = "--",color = "black")
# ax6.set_title("1 to 1")
# ax6.legend()
# ax6.set_ylim(0,200)
# ax6.set_xlim(0,200)
# plt.xlabel("Methane Reference [ppm]")
# plt.ylabel("Predicted Methane [ppm]")
# fig6.autofmt_xdate()

# # Calc RMSE 
# def RMSE(Actual,Predict):
#     n = len(Predict)
#     Sub = (Predict - Actual)**2
#     RMSE_calc = np.sqrt(np.sum(Sub)/n)
#     return RMSE_calc

# RMSE_Field_Calc = RMSE(LICOR_2["CH4"], Methane_2["CH4"])
# print(RMSE_Field_Calc)

#%% Export DATA
#Format: index = Timestamp, CH4, aTemp, B_Temp, B_Pressure, aFlow, RMSE
methane["Box Concentration"] = methane["CH4"]
methane.drop(["CH4"],axis=1,inplace = True)
methane["B_RMSE"] = methane["RMSE"]
methane["B_Pressure"] = DATA["BO Pressure (hPa)"]
#methane["aFlow"]= DATA["aFlow"]
methane["Ambient"] = A_CH4["A_CH4"]
methane["A_RMSE"] = A_RMS
methane["A_Pressure"] = DATA["Pressure (hPa)"]
methane["A_TEMP"] = DATA["FlowTemp (C)"]
methane["Temperature"] = DATA["FlowTemp (C)"]
methane["Flow Rate"] = DATA["FlowRate (slm)"]

#%%
'''
fig6,ax6 = plt.subplots(dpi=300)
ax6.hist(methane['B_TEMP'], bins=30, color='skyblue', edgecolor='black',label = "Field Temperatures")
ax6.hist([5,25,35], bins=30, color='yellow', edgecolor='black',label = "Calibration Temperatures")
ax6.legend()
ax6.set_xlabel("Temperature [C]")
ax6.set_ylabel("# of Occurances")

fig7,ax7 = plt.subplots(dpi=300)
ax7.hist(DATA['B_Humidity'], bins=30, color='orange',label = "Field Humidities")
ax7.hist([2,20,40,70], bins=30, color='yellow', edgecolor='black',label = "Calibration Humidities")
ax7.legend()
ax7.set_xlabel("Relative Humidity [%]")
ax7.set_ylabel("# of Occurances")
fig1.savefig("Figures/MethaneConc"+date+".png")


#Correct outputted Data for flux code
methane['Box Concentration'] = methane["B_CH4"]
methane["Ambient"] = methane["A_CH4"]
methane["Temperature"] = methane["B_TEMP"]
methane["Flow Rate"]=methane["aFlow"]

methane.drop(["A_CH4","B_TEMP","aFlow","B_CH4"],axis=1,inplace = True)
'''
#%% Save DATA
# methane.to_csv("Box_Concentrations/BOX_Concentration-"+date+".csv")
# print("\nData Exported \n")
#%% PDF Report

# pdf = FPDF()
# pdf.add_page()
# pdf.set_xy(0, 0)  
# pdf.set_font('arial', 'B', 14)
# pdf.cell(60)
# pdf.cell(100, 10, "Methane Quantification Report", 0, 2, 'C')
# pdf.set_font('arial',"U", 12)
# pdf.cell(100,10,"Methane Responce Internal Sensors [ppm]",0,2,'C')
# pdf.image("Figures/MethaneConc"+date+".png", x = 10, y = None, w = 200, h = 150, type = '', link = '')
# pdf.set_font('arial',"", 12)
# #pdf.cell(100,10,"RMSE or Error for Regression is %0.4f [ppm]"%(RMS),0,3,'L')
# pdf.cell(100,10,"Data from "+str(methane.index[0])+" to "+str(methane.index[len(methane)-1]),0,3,'C')
# #pdf.cell(100,10,"Maximum Concentration = %0.4f [ppm]"%(max(methane["Box Concentration"])),0,3,'C')
# #pdf.cell(100,10,"Minimum Concentration = %0.4f [ppm]"%(min(methane["Box Concentration"])),0,3,'C')



# pdf.output('Reports/'+date+'-MethaneReport.pdf', 'F')
# print("\nReport Generated\n")

