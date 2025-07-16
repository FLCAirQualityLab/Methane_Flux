# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 08:07:55 2025
Methane Group Data Wrangling
Takes in SGX, Soracom and setpoints and get averaged data
@author: Sam Hill

----------Please Read----------

DESCRIPTION
This code is used to generate averaged setpoint data from physical calibration. use the exported averaged data for the calibration code.

INPUT:
    Setpoint file .xlsx, this file should consist one one column named "Setpoint, one named "Start Time" and one named "Steady State Time"
        Both Start Time and Steady State Time should have when the setpoint started, and when it was done or at a steady state, this time should
        be in YYYY-MM-DD HH-MM-SS format. 
        
    USER INPUTS
    Visualization:
        Start Time: Start of setpoint visualization
        End Time: End of setpoint visualization
    
        
OUTPUT:
    Averaged Setpoint File named Averaged Setpoint+date.xlsx
"""

import pandas as pd
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import copy as cp
import gspread
import numpy as np
#%%Lucien Code


import sys
#path = 'C:/Users/lucie/OneDrive/Desktop/Boronius/Python/sheets_puller.py'

################################################
#-----------------INPUT------------------------#
#---------------Time Average-------------------#       
Start_Time = pd.to_datetime("2025-04-16 18:00:00")
End_Time = pd.to_datetime("2025-04-16 22:00:00")
SS_start = 23 #min
SS_end   = 27 #min                 #
sheet = "DATA_FULL_SWEEP-4-14" #change sheet name. case sensitive (data00 wont work, Data00 will)
################################################

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

def list_list(sheet):
    worksheet = sh.worksheet(sheet)
    lst = worksheet.get_all_values()
    return lst

def list_dict(sheet):
    worksheet = sh.worksheet(sheet)
    lst = worksheet.get_all_records()
    return lst

def sheet_dataframe(sheet):
    worksheet = sh.worksheet(sheet)
    dataframe = pd.DataFrame(worksheet.get_all_records())
    return dataframe
MOX = sheet_dataframe(sheet)
MOX1 = MOX
#%%get files

#get Data
root = tk.Tk()
root.withdraw()
root.lift()
root.attributes('-topmost', True)
#print("SORACOM FILE (.CSV)")
#MOX = filedialog.askopenfile(title = "Soracom Data",filetypes=(("CSV files","*.csv"),("all files", "*.*")))

print("Setpoint file (.XSLX)")
Setpoints = filedialog.askopenfile(title = "Setpoints",filetypes=(("CSV files","*.csv"),("all files", "*.*")))

#date = input("Todays Date: ")
date="4-14-SWEEP-2"
#MOX = pd.read_csv(MOX.name)
Setpoints = pd.read_csv(Setpoints.name)
    #%%




MOX = MOX.set_index(pd.to_datetime(MOX["Time"]))#.dt.tz_localize('MST'))
# Time average dATA
Setpoints["time"] = pd.to_datetime(Setpoints["time"])#.dt.tz_localize('MST')
MOX.drop(["Time","Main TGS2600 (mV)","Main TGS2602 (mV)","Main TGS2611 (mV)","Main EC_Worker (mV)","Main EC_Aux (mV)","Temperature (C)","Pressure (hPa)","Humidity (%)","GasResistance (Ohms)"],axis=1,inplace=True)
MOX.dropna(axis=0, inplace=True)
Setpoints.dropna(axis=0,inplace=True)
for i in range(len(MOX)):
    if MOX["SGX_Digital (ppm)"][i] == 404:
        MOX["SGX_Digital (ppm)"][i] = np.nan
MOX.dropna(axis=0,inplace=True)
MOX = MOX.resample("10s").last()
MOX.dropna(axis=0,inplace=True)
#%%

fig1,ax1 = plt.subplots(figsize=(10,5))

ax1.plot(MOX.index,MOX["BO TGS2602 (mV)"],label = "TGS02")
ax1.plot(MOX.index,MOX["BO TGS2611 (mV)"],label = "TGS11")
ax1.plot(MOX.index,MOX["BO TGS2600 (mV)"],label = "TGS00")
ax1.plot(MOX.index,MOX["SGX_Digital (ppm)"],label = "SGX")
for i in range(len(Setpoints)):
    ax1.axvline(x=Setpoints["time"][i],color = "black",linewidth=0.3)
    ax1.axvspan(Setpoints.iloc[i,6]+timedelta(minutes=SS_start),Setpoints.iloc[i,6]+timedelta(minutes=SS_end),alpha=0.5)
ax1.legend()
ax1.set_xlabel("Time")
ax1.set_ylabel("TGS2611 Responce [mV]")
ax1.set_xlim([Start_Time,End_Time])


#%%
setpointavg = pd.DataFrame(columns=[["BO TGS2600 (mV)","BO TGS2611 (mV)","BO TGS2602 (mV)","BO EC_Worker (mV)","BO EC_Aux (mV)","SGX_Analog (mV)","SGX_Digital (ppm)","BO Temperature (C)","BO Pressure (hPa)","BO Humidity (%)","BO GasResistance (Ohms)"]])

for i in range(len(Setpoints)):
    try:
        temp = cp.deepcopy(MOX)
        mean = temp.loc[Setpoints.iloc[i,6]+timedelta(minutes=SS_start):Setpoints.iloc[i,6]+timedelta(minutes=SS_end)]
        mean = mean.mean()
        setpointavg.loc[i] = mean.values
    except:
        print(" No Data %0.0f"%(i))
        setpointavg.loc[i] = [0,0,0,0,0,0,0,0,0,0,0]
#print(mean.values)
#print(mean)

    
setpointavg["Setpoint"] = Setpoints["C_CH4 [ppm]"]
#%%
setpointavg["H2S"] = Setpoints["C_H2S [ppm]"]
#%%
setpointavg.to_excel("Averaged_Setpoints"+str(date)+".xlsx")
print("Sucess!! File Created")
time.sleep(5) # wait 5 seconds

#%%
fig2,ax2 = plt.subplots()
ax2.plot(MOX.index,MOX["SGX_Digital (ppm)"],label = "SGX")
ax2.set_ylim(0,2000)
ax2.set_xlim([pd.to_datetime("2025-04-16 12:00:00"),pd.to_datetime("2025-04-16 22:00:00")])