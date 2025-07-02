"""
Methane Quantification Code V1
Jessica Goff
06.03.25
"""

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tkinter.filedialog as filedialog
from Python.sheets_puller import sheetPuller
# First-time users may need to install fpdf on device before importing
from fpdf import FPDF

"""
----------------------------------------------------------------- DESCRIPTION -----------------------------------------------------------------
This code is used to quantify methane data from  The resulting BOX concentrations should be input directly into the flux quantification code.

INPUT:
    Regression Models: lowreg.plk , midreg.plk, highreg.plk, Ambient_Colocate_reg.plk all saved in the directory /Coefficients
    RMSE error: B_RMSE.txt and A_RMSE.txt
        
    USER INPUTS
        licPrompt: Optional prompt to compare LICOR data
        licPath: Default path for LICOR data if not prompted
        sheet: Sheet within the Google Sheet to draw data from
        date: Used to append the generated CSV
        startTime: Used for x-axis lower bound in plotting
        endTime: Used for x-axis upper bound in plotting
    
OUTPUT:
    Box concentration file named 'BOX_Concentration-"+date+".csv', and 'MethaneReport-"+date+".pdf'
"""


#%% ---------------------------------------------------- USER INPUTS ----------------------------------------------------
licPrompt = False                                           # Do you want to be prompted to find your own file?
licPath = "LICOR.csv"                                       # default path for LICOR data unless otherwise specified
startTime = pd.to_datetime("2025-06-16 15:23:00")           # Must be within selected dataset
endTime = pd.to_datetime("2025-06-30 10:00:00")             # Must be in [yyyy-mm-dd hh-mm-ss] format
sheet = "Data"                                              # Sheet name to draw data from. Case sensitive.

pcbNum = 4                                                  # What Boron are you using?

# Calling sheetPuller function
DATA = sheetPuller(sheet, f"Boron_{pcbNum}")


#%% Importing LICOR data (optional)...
def csvPrompt(prompt = False, default = "", title = "Select LICOR (.csv) file..."):
    if prompt == True:
        file = filedialog.askopenfile(title=title, filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if file is None:
            raise FileNotFoundError("No file selected.")
        return pd.read_csv(file.name)
    else:
        return pd.read_csv(licPath)

if licPrompt == True:
    print("Loading LICOR data...")
    LICOR = csvPrompt(prompt = licPrompt, default = licPath, title = "Select LICOR (.csv) file...")
    LICOR["DateTime"] = pd.to_datetime(LICOR['DATE'] + ' ' + LICOR['TIME'])
    LICOR.set_index("DateTime", inplace=True)
    LICOR.drop(columns=[
       "DATAH", "SECONDS", "NANOSECONDS", "NDX", "DIAG", "REMARK", "DATE", "TIME",
       "H2O", "CO2", "RESIDUAL", "RING_DOWN_TIME", "THERMAL_ENCLOSURE_T", "PHASE_ERROR",
       "LASER_T_SHIFT", "INPUT_VOLTAGE", "CHK", "CAVITY_P", "CAVITY_T", "LASER_PHASE_P", "LASER_T"
       ], inplace=True, errors='ignore')
    LICOR.dropna(inplace=True)


#%% DATA data preprocessing...
# Converting time to datetime and setting as index
DATA["Time"] = pd.to_datetime(DATA["Time"])
DATA.set_index("Time", inplace=True)
DATA.sort_index(inplace=True)                   # Ensure time is sorted properly

# Removing rows with the boot-up timestamp
DATA = DATA[DATA.index != pd.Timestamp("1999-12-31 17:00:12")]

# Replacing error codes and dropping missing data
DATA.replace(404, np.nan, inplace=True)
DATA.dropna(inplace=True)

# Filtering to the date range of interest
print(f"\n---- DATA Adjustments ----\nDATA size before filtering: {len(DATA)}")
DATA = DATA[(DATA.index >= startTime) & (DATA.index <= endTime)]
print(f"DATA size after filtering: {len(DATA)}\n")

# Time-averaging the data
DATA = DATA.resample("60s").last().dropna()

# Dropping rows with impossible pressures and gas resistances to prevent dividebyzero error
DATA = DATA[DATA["Pressure (hPa)"] != 0]
DATA = DATA[DATA["GasResistance (Ohms)"] != 0]
DATA = DATA[DATA["Humidity (%)"] != 0]


#%% Importing regressors and RMSE files...
# Loading regression models
def loadModel(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# Loading .txt RMSE files
def loadRmse(path):
    with open(path, "r") as f:
        return float(f.read())

# Calling loadModel to retrieve regression models
reg_low = loadModel(f'Coefficients/lowreg_Boron{pcbNum}.pkl')
reg_mid = loadModel(f'Coefficients/midreg_Boron{pcbNum}.pkl')
reg_high = loadModel(f'Coefficients/highreg_Boron{pcbNum}.pkl')
reg_amb = loadModel('Coefficients/Ambient_Colocate_reg.pkl')
# Calling loadRmse to retrieve RMSE files
RMSlow = loadRmse(f'Coefficients/B_RMSE_L_Boron{pcbNum}.txt')
RMSmid = loadRmse(f'Coefficients/B_RMSE_M_Boron{pcbNum}.txt')
RMShigh = loadRmse(f'Coefficients/B_RMSE_H_Boron{pcbNum}.txt')
#A_RMS = loadRmse('Coefficients/A_RMSE.txt')

# Creating timestamps from index (datetime from DATA [122])
TS = DATA.index


#%% Ambient regression...

# # Creating A_Coef dataframe and renaming columns
# A_COEF = DATA[[
#     'Main TGS2600 (mV)',
#     'Main TGS2611 (mV)',
#     'Main TGS2602 (mV)',
#     'Main EC_Worker (mV)',
#     'Temperature (C)',
#     'Pressure (hPa)',
#     'Humidity (%)'
# ]
# ].copy()

# A_COEF.rename(columns = {
#     'Main TGS2600 (mV)': 'TGS2600',
#     'Main TGS2611 (mV)': 'TGS2611',
#     'Main TGS2602 (mV)': 'TGS2602',
#     'Main EC_Worker (mV)': 'EC',
#     'Temperature (C)': 'Temp',
#     'Pressure (hPa)': 'Pressure',
#     'Humidity (%)': 'Humidity'
# }, inplace = True)

# # Ensuring column order and names are identical to the training set
# A_COEF = A_COEF[reg_low.feature_names_in_]

# # Generating ambient predictions using the 'reg_low' model
# A_CH4 = pd.DataFrame({"A_CH4": reg_low.predict(A_COEF)}, index = TS)

#%% Ambient regression...

# 1. Create your new ratio features in the main DATA DataFrame
DATA["R1"] = DATA["Main TGS2611 (mV)"] / DATA["Main TGS2600 (mV)"]
DATA["R2"] = DATA["Temperature (C)"] / DATA["Humidity (%)"]
DATA["R3"] = DATA["Main TGS2602 (mV)"] / DATA["GasResistance (Ohms)"]
DATA["R4"] = DATA["Temperature (C)"] / DATA["Main TGS2602 (mV)"]

# 2. Select and rename the base ambient features into a new DataFrame
A_COEF_base = DATA[[
    "GasResistance (Ohms)", "Humidity (%)", "Pressure (hPa)", "Temperature (C)",
    "Main TGS2600 (mV)", "Main TGS2602 (mV)", "Main TGS2611 (mV)",
]].rename(columns={
    "GasResistance (Ohms)": "A_Gas",
    "Humidity (%)": "A_Humidity",
    "Pressure (hPa)": "A_Pressure",
    "Temperature (C)": "A_TEMP",
    "Main TGS2600 (mV)": "A_TGS00",
    "Main TGS2602 (mV)": "A_TGS02",
    "Main TGS2611 (mV)": "A_TGS11"
})

# 3. Combine the base features and ratio features into the final feature set
# We use .join() because they share the same index
A_COEF = A_COEF_base.join(DATA[["R1", "R2", "R3", "R4"]])

# 4. Remove any rows with missing values and predict
A_COEF.dropna(inplace=True)
A_CH4 = pd.DataFrame({"A_CH4": reg_amb.predict(A_COEF)}, index=A_COEF.index)

# --- Optional: Print statistics ---
print(f"The maximum ambient methane concentration is {np.round(np.max(A_CH4['A_CH4']), 4)} [ppm].")
print(f"The minimum ambient methane concentration is {np.round(np.min(A_CH4['A_CH4']), 4)} [ppm].")
print(f"The average ambient methane concentration is {np.round(np.mean(A_CH4['A_CH4']), 4)} [ppm].")


#%% Building feature sets...
# Creating low coefficient dataframe features
low_model_features = [
    'BO TGS2600 (mV)', 'BO TGS2602 (mV)', 
    'BO Temperature (C)', 'BO Humidity (%)'
]

# Creating medium dataframe features with low features + SGX
med_model_features = low_model_features + ['SGX_Digital (ppm)']

# Creating high coefficient dataframe features
high_model_features = [
    'SGX_Digital (ppm)', 'BO Temperature (C)', 'BO Humidity (%)'
]

# Create the initial feature DataFrames by selecting columns from DATA
COEF_L = DATA[low_model_features].copy()
COEF_M = DATA[med_model_features].copy()
COEF_H = DATA[high_model_features].copy()


#%% Renaming columns to match model expectations...
# Renaming columns for all coefficient dataframes
rename_map = {
    'BO TGS2600 (mV)': 'TGS2600',
    'BO TGS2611 (mV)': 'TGS2611',
    'BO TGS2602 (mV)': 'TGS2602',
    'BO EC_Worker (mV)': 'EC',
    'BO Temperature (C)': 'Temp',
    'BO Humidity (%)': 'Humidity',
    'SGX_Digital (ppm)': 'SGX_Digital'
}


# Apply this renaming to all three of your feature DataFrames.
COEF_L.rename(columns=rename_map, inplace=True)
COEF_M.rename(columns=rename_map, inplace=True)
COEF_H.rename(columns=rename_map, inplace=True)

# Ensuring column names are the same as the models and in the same order
COEF_L = COEF_L[reg_low.feature_names_in_]
COEF_M = COEF_M[reg_mid.feature_names_in_]
COEF_H = COEF_H[reg_high.feature_names_in_]


#%% Applying regressions...
low = pd.DataFrame({"CH4": reg_low.predict(COEF_L)}, index=TS)
mid = pd.DataFrame({"CH4": reg_mid.predict(COEF_M)}, index=TS)
high = pd.DataFrame({"CH4": reg_high.predict(COEF_H)}, index=TS)

low["RMSE"] = RMSlow
mid["RMSE"] = RMSmid
high["RMSE"] = RMShigh

# Calibration Offsets
# low["CH4"] += 310
# mid["CH4"] -= 1325

# Making a copy of mid regression called "methane"
methane = mid.copy() * 0

# Splitting up the CH4 values and associated rows into low, mid, and high
for i in range(len(methane)):
    val_low, val_mid, val_high = low.iloc[i]["CH4"], mid.iloc[i]["CH4"], high.iloc[i]["CH4"]

    if 2 <= val_low < 75:
        methane.iloc[i] = low.iloc[i]
    elif 75 <= val_mid < 700:
        methane.iloc[i] = mid.iloc[i]
    elif 700 <= val_high <= 11000:
        methane.iloc[i] = high.iloc[i]
    else:
        methane.iloc[i] = [np.nan, np.nan]

methane.dropna(inplace=True)


#%% Creating plots...
fig1,ax1 = plt.subplots(dpi=300)
ax1.plot(methane.index,methane["CH4"], color= "orange",label = "Sensor Array")
ax1.fill_between(methane.index,methane["CH4"]+methane["RMSE"],methane["CH4"]-methane["RMSE"], alpha=0.4, color = "orange")
#if Lic == "T":
    #ax1.plot(LICOR.index,LICOR["CH4"]/1000,label = "Reference Instrument",color="red")
ax1.legend()
ax1.set_xlabel("Time [DD HH:MM]")
ax1.set_ylabel("Methane [ppm]")
fig1.autofmt_xdate()
#ax1.set_ylim(-2,200)
ax1.set_xlim(startTime,endTime)
fig1.savefig(f"Figures/MethaneConcBoron{pcbNum}.png")

fig2,ax2 = plt.subplots(dpi=300)
ax2.plot(low.index,low["CH4"])
ax2.set_title("Low Reg CH4")
fig2.autofmt_xdate()
#ax2.set_ylim(0,100)
ax2.set_xlim(startTime,endTime)

fig3,ax3 = plt.subplots(dpi=300)
ax3.plot(mid.index,mid["CH4"])
ax3.set_title("Mid Reg CH4")
fig3.autofmt_xdate()
#ax3.set_ylim(75,1000)
ax3.set_xlim(startTime,endTime)

fig4,ax4 = plt.subplots(dpi=300)
ax4.plot(high.index,high["CH4"])
ax4.set_title("High Reg CH4")
fig4.autofmt_xdate()
#ax4.set_ylim(500,11000)
ax4.set_xlim(startTime,endTime)

fig5,ax5 = plt.subplots(dpi=300)
ax5.plot(DATA.index,DATA["BO TGS2600 (mV)"],label = "TGS00")
ax5.plot(DATA.index,DATA["BO TGS2602 (mV)"],label = "TGS02")
ax5.plot(DATA.index,DATA["BO TGS2611 (mV)"],label = "TGS11")
ax5.set_title("MOX Sensor Readings")
ax5.legend()
fig5.autofmt_xdate()
ax5.set_xlim(startTime,endTime)


#%% Exporting DATA...
#Format: index = Timestamp, CH4, aTemp, B_Temp, B_Pressure, aFlow, RMSE
methane["Box Concentration"] = methane["CH4"]
methane.drop(["CH4"],axis = 1, inplace = True)
methane["B_RMSE"] = methane["RMSE"]
#methane["B_Pressure"] = DATA["BO Pressure (hPa)"]
#methane["aFlow"]= DATA["aFlow"]
methane["Ambient"] = A_CH4["A_CH4"]
#methane["A_RMSE"] = A_RMS
#methane["A_Pressure"] = DATA["Pressure (hPa)"]
methane["A_TEMP"] = DATA["FlowTemp (C)"]
methane["Temperature"] = DATA["FlowTemp (C)"]
methane["Flow Rate"] = DATA["FlowRate (slm)"]


#%% Exporting data...
methane.to_csv(f"Box_Concentrations/boxConcentration_Boron{pcbNum}.csv")
print("\nBox Concentration CSV exported :)\n")


#%% Generating PDF Report...

pdf = FPDF()
pdf.add_page()
pdf.set_xy(0, 0)  
pdf.set_font('arial', 'B', 14)
pdf.cell(60)
pdf.cell(100, 10, "Methane Quantification Report", 0, 2, 'C')
pdf.set_font('arial',"U", 12)
pdf.cell(100,10,"Methane Response Internal Sensors [ppm]",0,2,'C')
pdf.image(f"Figures/MethaneConcBoron{pcbNum}.png", x = 10, y = None, w = 200, h = 150, type = '', link = '')
pdf.set_font('arial',"", 12)
pdf.cell(100,10,"Data from "+str(methane.index[0])+" to "+str(methane.index[len(methane)-1]),0,3,'C')
pdf.cell(100,10,"Maximum Concentration = %0.4f [ppm]"%(max(methane["Box Concentration"])),0,3,'C')
pdf.cell(100,10,"Minimum Concentration = %0.4f [ppm]"%(min(methane["Box Concentration"])),0,3,'C')

pdf.output(f'Reports/MethaneReportBoron{pcbNum}.pdf', 'F')
print("\nMethane report generated!\n")



#%% testomg
fig1, ax1 = plt.subplots(1,1, dpi =600)
ax1.plot(methane.index, methane["Ambient"], label = 'Ambient')
ax1.plot(methane.index, methane["Box Concentration"], label = "Internal")
ax1.legend()
ax1.tick_params(axis='x', labelrotation=45)
for label in ax1.get_xticklabels():
    label.set_ha('right')
ax1.set_xlabel("Time")
ax1.set_ylabel("Methane Concentration (ppm)")
ax1.set_title("Ambient and Internal Methane Concentrations")

#ax.plot(TS, A_COEF["TGS2611"], label="Amb")
#ax.plot(TS, DATA["BO TGS2611 (mV)"], label="Internal")
#ax.plot(TS, A_COEF["Temp"], label="Amb")
#ax.plot(TS, DATA["BO Temperature (C)"], label="Internal")
#ax.plot(TS, A_COEF["Humidity"], label="Amb")
#ax.plot(TS, DATA["BO Humidity (%)"], label="Internal")
#ax.plot(TS, A_COEF["TGS2602"], label="Amb")
#ax.plot(TS, DATA["BO TGS2602 (mV)"], label="Internal")
# ax.plot(TS, A_COEF["TGS2600"], label="Amb")
# ax.plot(TS, DATA["BO TGS2600 (mV)"], label="Internal")
# ax.legend()