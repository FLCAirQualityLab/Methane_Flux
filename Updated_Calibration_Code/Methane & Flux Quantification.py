"""
Methane & FLux Quantification Code
Jessica Goff
06.03.25
"""

# Importing loading UI essentials...
from Python.loadingUI import startLoading, stopLoading
lE, lT = startLoading(message = "Importing libraries")

# Importing libraries
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pickle
    from pathlib import Path
    import tkinter.filedialog as filedialog
    from Python.sheets_puller import sheetPuller
    # First-time users may need to install fpdf on device before importing
    from fpdf import FPDF
except Exception as e:
    stopLoading(lE, lT, done=False)
    raise e
stopLoading(lE, lT)

"""
----------------------------------------------------------------- DESCRIPTION -----------------------------------------------------------------
This code is used to quantify methane data from  The resulting BOX concentrations should be input directly into the flux quantification code.

INPUT:
    Regression Models: lowreg.plk , midreg.plk, highreg.plk, ambreg.plk all saved in the directory /Coefficients
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

# -------------------------------------------------------------- Flux Quantification code begins on line 433 -----------------------------------------------------------

#%% ---------------------------------------------------- USER INPUTS ----------------------------------------------------
# Importing LICOR file
licPrompt = False                                           # Do you want to be prompted to find your own file?

licPath = "LICOR.csv"                                       # default path for LICOR data unless otherwise specified

startTime = pd.to_datetime("2025-06-30 00:00:00")           # Must be within selected dataset
endTime = pd.to_datetime("2025-07-05 00:00:00")             # Must be in [yyyy-mm-dd hh-mm-ss] format

pcbNum = 9                                                  # What Boron are you using?

lE, lT = startLoading(message = "Importing data", t=0.25)
# Calling sheetPuller function
try: DATA = sheetPuller("Data", f"Boron_{pcbNum}")
except Exception as e:
    stopLoading(lE, lT, done=False)
    print("\nAn error was encountered while importing data!")
    raise e
stopLoading(lE, lT)

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

lE, lT = startLoading(message = "Cleaning data", t=0.25)

try:
    DATA = DATA[DATA['Time'] != 'Time']
    # Converting time to datetime and setting as index
    DATA["Time"] = pd.to_datetime(DATA["Time"], format = 'mixed')
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
    
    # Creating feature interactions and ratios
    DATA["Temp*Humid"] = DATA["BO Temperature (C)"] * DATA["BO Humidity (%)"]
    DATA["R11_00"] = DATA["BO TGS2611 (mV)"] / DATA["BO TGS2600 (mV)"]
    DATA["2611*Temp"] = DATA["BO TGS2611 (mV)"] * DATA["BO Temperature (C)"]
    DATA["SGX*Temp"] = DATA["SGX_Digital (ppm)"] * DATA["BO Temperature (C)"]

except Exception as e:
    stopLoading(lE, lT, done=False)
    raise e

stopLoading(lE, lT)

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
reg_amb = loadModel(f'Coefficients/ambreg_Boron{pcbNum}.pkl')
# Calling loadRmse to retrieve RMSE files
RMSlow = loadRmse(f'Coefficients/RMSE_low_Boron{pcbNum}.txt')
RMSmid = loadRmse(f'Coefficients/RMSE_med_Boron{pcbNum}.txt')
RMShigh = loadRmse(f'Coefficients/RMSE_high_Boron{pcbNum}.txt')
A_RMS = loadRmse(f'Coefficients/RMSE_amb_Boron{pcbNum}.txt')

# Creating timestamps from index (datetime from DATA [116])
TS = DATA.index

#%% Ambient regression...

# Creating A_Coef dataframe and renaming columns
A_COEF = DATA[[
    'Main TGS2600 (mV)',
    'Main TGS2611 (mV)',
    'Main TGS2602 (mV)',
    'Main EC_Worker (mV)',
    'Temperature (C)',
    'Pressure (hPa)',
    'Humidity (%)'
]].copy()

# Renaming columns to match the model's expectations
A_COEF.rename(columns = {
    'Main TGS2600 (mV)': 'aTGS2600',
    'Main TGS2611 (mV)': 'aTGS2611',
    'Main TGS2602 (mV)': 'aTGS2602',
    'Main EC_Worker (mV)': 'aEC',
    'Temperature (C)': 'aTemp',
    'Pressure (hPa)': 'aPressure',
    'Humidity (%)': 'aHumidity'
}, inplace = True)

# Adding in the missing interaction features using the NEW column names
A_COEF['aTemp*Humid'] = A_COEF['aTemp'] * A_COEF['aHumidity']
A_COEF['aR11_00'] = A_COEF['aTGS2611'] / A_COEF['aTGS2600']

# Ensuring column order and names are identical to the training set
A_COEF = A_COEF[reg_amb.feature_names_in_]

# Generating ambient predictions using the 'reg_low' model
A_CH4 = pd.DataFrame({"A_CH4": reg_amb.predict(A_COEF)}, index = TS)

#%% Building feature sets...

# Creating low coefficient dataframe features
low_model_features = ['BO TGS2600 (mV)', 'BO TGS2611 (mV)', 'R11_00', 'Temp*Humid']

# Creating medium dataframe features with low features + SGX
med_model_features = ['BO TGS2600 (mV)', 'BO TGS2611 (mV)', 'Temp*Humid', '2611*Temp', 'BO Humidity (%)']

# Creating high coefficient dataframe features
high_model_features = ['BO TGS2611 (mV)', 'SGX_Digital (ppm)', 'BO Temperature (C)', 'BO Humidity (%)', '2611*Temp', 'SGX*Temp']

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

# Predict concentrations using each model
low = pd.DataFrame({"CH4": reg_low.predict(COEF_L)}, index=TS)
mid = pd.DataFrame({"CH4": reg_mid.predict(COEF_M)}, index=TS)
high = pd.DataFrame({"CH4": reg_high.predict(COEF_H)}, index=TS)

# Assign the corresponding RMSE to each prediction set
low["RMSE"] = RMSlow
mid["RMSE"] = RMSmid
high["RMSE"] = RMShigh


# Sorting into ranges for methane concentrations
conditions = [
    (low['CH4'] < 75),
    (mid['CH4'] >= 75) & (mid['CH4'] < 700),
    (high['CH4'] >= 700) & (high['CH4'] <= 11000),
]

# 2. Define the choices for the 'CH4' column that correspond to each condition.
ch4_choices = [
    low['CH4'],
    mid['CH4'],
    high['CH4'],
]

# 3. Define the choices for the 'RMSE' column.
rmse_choices = [
    low['RMSE'],
    mid['RMSE'],
    high['RMSE'],
]

#    Building the 'methane' DataFrame
#    It picks the choice corresponding to the first true condition
#    If no condition is true for a row, it uses the 'default' value (np.nan)
methane = pd.DataFrame(index=TS)
methane['CH4'] = np.select(conditions, ch4_choices, default=np.nan)
methane['RMSE'] = np.select(conditions, rmse_choices, default=np.nan)

#    Dropping the rows where no model was deemed valid
methane.dropna(inplace=True)

print(f"Successfully generated methane predictions. Final DataFrame size: {len(methane)}")
#%% Creating plots...
lE, lT = startLoading(message = "Generating Figures")

try:
    fig1,ax1 = plt.subplots(dpi = 600)
    ax1.plot(methane.index,methane["CH4"], color= "orange",label = "Sensor Array")
    ax1.fill_between(methane.index,methane["CH4"]+methane["RMSE"],methane["CH4"]-methane["RMSE"], alpha=0.4, color = "orange")
    #if Lic == "T":
        #ax1.plot(LICOR.index,LICOR["CH4"]/1000,label = "Reference Instrument",color="red")
    ax1.legend()
    ax1.set_xlabel("Time [DD HH:MM]")
    ax1.set_ylabel("Methane [ppm]")
    fig1.autofmt_xdate()
    ax1.set_xlim(startTime,endTime)
    fig1.savefig(f"Figures/MethaneConcBoron{pcbNum}.png")
    
    fig2,ax2 = plt.subplots(dpi=300)
    ax2.plot(low.index,low["CH4"])
    ax2.set_title("Low Reg CH4")
    fig2.autofmt_xdate()
    ax2.set_xlim(startTime,endTime)
    
    fig3,ax3 = plt.subplots(dpi=300)
    ax3.plot(mid.index,mid["CH4"])
    ax3.set_title("Mid Reg CH4")
    fig3.autofmt_xdate()
    ax3.set_xlim(startTime,endTime)
    
    fig4,ax4 = plt.subplots(dpi=300)
    ax4.plot(high.index,high["CH4"])
    ax4.set_title("High Reg CH4")
    fig4.autofmt_xdate()
    ax4.set_xlim(startTime,endTime)
    
    fig5,ax5 = plt.subplots(dpi = 600)
    ax5.plot(DATA.index,DATA["BO TGS2600 (mV)"],label = "TGS00")
    ax5.plot(DATA.index,DATA["BO TGS2602 (mV)"],label = "TGS02")
    ax5.plot(DATA.index,DATA["BO TGS2611 (mV)"],label = "TGS11")
    ax5.set_title("MOX Sensor Readings")
    ax5.legend()
    fig5.autofmt_xdate()
    ax5.set_xlim(startTime,endTime)
    
except Exception as e:
    stopLoading(lE, lT, done=False)
    print("\n An error was encountered while generating figures!")
    raise e

stopLoading(lE, lT)

#%% Exporting DATA...
lE, lT = startLoading(message = "Exporting Data & Figures")

try:
    #Format: index = Timestamp, CH4, aTemp, B_Temp, B_Pressure, aFlow, RMSE
    methane["Box Concentration"] = methane["CH4"]
    methane.drop(["CH4"],axis = 1, inplace = True)
    methane["B_RMSE"] = methane["RMSE"]
    methane["B_Pressure"] = DATA["BO Pressure (hPa)"]
    methane["Ambient"] = A_CH4["A_CH4"]
    methane["A_RMSE"] = A_RMS
    methane["A_Pressure"] = DATA["Pressure (hPa)"]
    methane["A_TEMP"] = DATA["FlowTemp (C)"]
    methane["Temperature"] = DATA["FlowTemp (C)"]
    methane["Flow Rate"] = DATA["FlowRate (slm)"]
    
    #%% Exporting data...
    methane.to_csv(f"Box_Concentrations/boxConcentration_Boron{pcbNum}polynomial.csv")
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
    
    pdf.output(f'Reports/MethaneReport_Boron{pcbNum}.pdf', 'F')
    print("\nMethane report generated!\n")
    
    #%% Plotting ambient and internal methane concentrations
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
    
    #%% Plotting comparison of data in the calibration vs the field via boxplot
    # What variable do you want to plot?
    plotVariable = "Box Concentration"
    
    # Setting times for calibration and field deployment
    TS = methane.index
    # Create a copy for the calibration data and add a 'Period' column
    cal = methane.loc[(TS >= startTime) & (TS <= pd.to_datetime("2025-06-03 23:23:00"))].copy()
    cal['Period'] = 'Calibration'
    
    # Create a copy for the field data and add a 'Period' column
    field = methane.loc[(TS >= pd.to_datetime("2025-06-16 15:17:00")) & (TS <= endTime)].copy()
    field['Period'] = 'Field'
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 6), dpi=600, sharey=False)
    
    # --- Plot 1: Calibration Data ---
    sns.boxplot(
        data=cal,
        y=plotVariable,
        ax=axs[0],           # Plot on the first axis
        color=sns.color_palette('Set2')[0],
        showfliers=False
    )
    axs[0].set_title('Calibration Period', fontsize=14)
    axs[0].set_ylabel('Methane Concentration $[ppm]$', fontsize=12)
    axs[0].grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
    
    
    # --- Plot 2: Field Data ---
    sns.boxplot(
        data=field,
        y=plotVariable,
        ax=axs[1],           # Plot on the second axis
        color=sns.color_palette('Set2')[1],
        showfliers=False
    )
    axs[1].set_title('Field Period', fontsize=14)
    axs[1].grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
    
    # Add a main title for the entire figure
    fig.suptitle(f"{plotVariable} Distribution", fontsize=16, y=0.98)
    
    # Adjust layout to prevent titles from overlapping
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust rect to make space for suptitle
    
    # Displaying the Final Plot ---
    plt.show()

except Exception as e:
    stopLoading(lE, lT, done=False)
    print("An error was encountered while exporting data!")
    raise e
    
stopLoading(lE, lT)


#%% --------------- FLUX QUANTIFICATION STARTS HERE ---------------------------
#%% User-defined variables...
intervalMins = 1        # Number of minutes over which to average methane values for flux calcs

# Ensure destination path exists!
Path("Flux Quantification").mkdir(parents=True, exist_ok=True)

#%% User Defined Functions
# Converting Celcius to Kelvin
c2K = lambda C: C + 273.15

# Converting [ppm] to [mg/m^3] (ignore error, it's ok :)
ppmMg = lambda pres, temp, conc: pres/(constants["R"]*temp)*constants["mW_mg"]*(conc/1000000)

# Calculating Area [m^2]
area = lambda D: (1/4)*np.pi*(D**2)

# Calculating Steady State Flux [g/day*m^2]
ssFlux = lambda Q, cB, cA, A: Q*(cB-cA)/A  # mg/s*m^2

# Conversion from [mg/s] to [g/day]
toGDay = lambda flux: flux*84.6

# Calculating Transient Flux [g/day*m^2]         1 Minute Time Intervals
transFlux = lambda Q, cB, cA, s, A, V: Q*(cB-cA)/A + s*V/A/60   # mg/s*m^2

# Calculating uncertinaty in flux
def uncSs(Q, uncQ, cB, uncCb, cA, uncCa, A):
    uncFlow = (((cB - cA) / A)**2) * (uncQ**2)
    uncBox = ((Q / A)**2) * (uncCb**2)
    uncAmb = ((-Q / A)**2) * (uncCa**2)
    return np.sqrt(uncFlow + uncBox + uncAmb)

def uncTrans(Q, uncQ, cB, uncCb, cA, uncCa, s, uncS, A, V):
    unc_ss_squared = uncSs(Q, uncQ, cB, uncCb, cA, uncCa, A)**2
    unc_slope_squared = (V / (A * 60))**2 * (uncS**2)
    return np.sqrt(unc_ss_squared + unc_slope_squared)

# Calculating std error of the mean
sem = lambda sd, n: sd/np.sqrt(n)

# Uncertianty propagation for addition/subtraction
uncProp = lambda x, y: np.sqrt((x)**2 + (y)**2)

#%% Reporting stats...
# Reporting mean and SD of ambient concentrations
print(f'\nThe mean ambient methane concentration is ~{np.round(np.mean(methane["Ambient"]),2)} [ppm].')
print(f'The standard deviation of the ambient concentration is ~{np.round(np.std(methane["Ambient"]),2)} [ppm].')

# Reporting mean and SD of box concentrations
print(f'\nThe mean box methane concentration is ~{np.round(np.mean(methane["Box Concentration"]),2)} [ppm].')
print(f'The standard deviation of the box concentration is ~{np.round(np.std(methane["Box Concentration"]),2)} [ppm].')

#%% Defining constants
constants = {"P":78900,
             "R":8.3145,
             "mW_mg":16043,
             "A": area(0.4699),
             "H": 0.111,
             "vol": 0.111*area(0.4699)}

#%% Time averaging data...
methaneAvg = methane.resample(f"{intervalMins}min").last().dropna(axis=0)

for col in methaneAvg:
    methaneAvg[col] = methaneAvg[col].astype(float)

#%% Converting averaged values to SI Units :)
# Temperature from [C] to [K]
methaneAvg["A_TEMP"]      = c2K(methaneAvg["A_TEMP"])
methaneAvg["Temperature"] = c2K(methaneAvg["Temperature"])

# Converting flow rate from [slm] to [m^3/s]
methaneAvg["Flow Rate"] = methaneAvg["Flow Rate"].astype(float)*(0.001/60)

# Concentrations (incl RMSE) from [ppm] to [mg/m^3]
methaneAvg["Box Concentration"] = ppmMg(methaneAvg["B_Pressure"],
                                        methaneAvg["Temperature"],
                                        methaneAvg["Box Concentration"]).astype(float)
methaneAvg["Ambient"] = ppmMg(methaneAvg["A_Pressure"], methaneAvg["A_TEMP"],      methaneAvg["Ambient"]).astype(float)
methaneAvg["B_RMSE"]  = ppmMg(methaneAvg["B_Pressure"], methaneAvg["Temperature"], methaneAvg["B_RMSE"]).astype(float)
methaneAvg["RMSE"]    = ppmMg(methaneAvg["A_Pressure"], methaneAvg["A_TEMP"],      methaneAvg["RMSE"]).astype(float)

#%% Calculating transient flux... ------ @RiverH ------
# We use a 3-point moving window for calculations. Change "3" to a different number to alter this.
# Calculating smoothed/running average box & amb concentrations
methaneAvg["cBSmooth"] = methaneAvg["Box Concentration"].rolling(3, center=True).mean()
methaneAvg["cASmooth"] = methaneAvg["Ambient"].rolling(3, center=True).mean()

# Calculate the rate of change (slope, s) of the smoothed box concentration (s = dC/dt)
#                     |------------ dC -----------| / |---------------------- dt --------------------------|
methaneAvg["slope"] = methaneAvg["cBSmooth"].diff() / methaneAvg.index.to_series().diff().dt.total_seconds() 

# You can't compute a slope using 1 point, so leading and trailing slope values are NaN. Therefore we must...
methaneAvg.dropna(inplace=True)
# /\ This will drop a few minutes of data, so make sure to pad your start and end times accordingly!
# Now we calculate transient flux [g/day*m^2] with freshly calculated smoothed values & slopes :D
methaneAvg["transFlux"] = toGDay(transFlux(methaneAvg["Flow Rate"], methaneAvg["cBSmooth"],
                                           methaneAvg["cASmooth"],methaneAvg["slope"],
                                           constants["A"], constants["vol"]))

#%% Calculating steady-state flux [g/day*m^2] (saved the easy part for last, yay very epic)
methaneAvg["ssFlux"] = toGDay(ssFlux(methaneAvg["Flow Rate"], methaneAvg["Box Concentration"],
                                     methaneAvg["Ambient"], constants["A"])).astype(float)

#%% Propagating error stats to flux values!
# Defining flow rate uncertianty... data sheet says worst unc is ~ 2.8%
uncQ  = 0.028*methaneAvg["Flow Rate"]
# Defining slope uncertianty ...
uncS = (np.sqrt(methaneAvg["B_RMSE"]**2 + methaneAvg["B_RMSE"].shift(1)**2) /\
        methaneAvg.index.to_series().diff().dt.total_seconds())

# Propagating uncertainties into final flux values :/
# Steady state [g/day*m^2]
methaneAvg['ssFluxUnc'] = toGDay(uncSs(Q     = methaneAvg["Flow Rate"],
                                       uncQ  = uncQ,
                                       cB    = methaneAvg["Box Concentration"],
                                       uncCb = methaneAvg["B_RMSE"],
                                       cA    = methaneAvg["Ambient"],
                                       uncCa = methaneAvg["RMSE"],
                                       A     = constants["A"]))
# Transient [g/day*m^2]
methaneAvg['transFluxUnc'] = toGDay(uncTrans(Q     = methaneAvg["Flow Rate"],
                                             uncQ  = uncQ,
                                             cB    = methaneAvg["cBSmooth"],
                                             uncCb = methaneAvg["B_RMSE"],
                                             cA    = methaneAvg["cASmooth"],
                                             uncCa = methaneAvg["RMSE"],
                                             A     = constants["A"],
                                             s     = methaneAvg["slope"],
                                             uncS  = uncS,
                                             V     = constants["vol"]))

methaneAvg.dropna(inplace=True)

del uncQ, uncS, constants

#%% Plotting flux results :)
lE, lT = startLoading(message="Generating and exporting flux figures")

try:
    # Function to average data over time
    tAvg = lambda data, mins: data.resample(f"{mins}min").mean()
    
    # Transient vs. steady state flux
    figA, axA = plt.subplots(1,1, dpi=600, figsize=(10,5))
    axA.plot(methaneAvg.index, methaneAvg["ssFlux"], label="Steady State")
    axA.plot(methaneAvg.index, methaneAvg["transFlux"], label="Transient")
    axA.fill_between(methaneAvg.index, methaneAvg["ssFlux"] + methaneAvg["ssFluxUnc"],
                     methaneAvg["ssFlux"] - methaneAvg["ssFluxUnc"],alpha = 0.4, color='steelblue')
    axA.fill_between(methaneAvg.index, methaneAvg["transFlux"] + methaneAvg["transFluxUnc"],
                     methaneAvg["transFlux"] - methaneAvg["transFluxUnc"], alpha = 0.4, color='orange')
    axA.legend()
    axA.set_xticks(axA.get_xticks(), axA.get_xticklabels(), rotation=45, ha='right')
    axA.set_title("Flux Rates", fontweight='bold')
    axA.set_xlabel("Time $[YYYY-DD-MM]$")
    axA.set_ylabel("Methane Flux $[\dfrac{g}{day}m^2]$")
    plt.show()
    
    # Transient vs. steady state flux w/ 1hr time averaging
    figB, axB = plt.subplots(1,1, dpi=600, figsize=(10,5))
    axB.plot(tAvg(methaneAvg, 60).index, tAvg(methaneAvg["ssFlux"], 60), label="Steady State")
    axB.plot(tAvg(methaneAvg, 60).index, tAvg(methaneAvg["transFlux"], 60), label="Transient")
    axB.fill_between(tAvg(methaneAvg, 60).index, tAvg(methaneAvg["ssFlux"] + methaneAvg["ssFluxUnc"], 60),
                     tAvg(methaneAvg["ssFlux"] - methaneAvg["ssFluxUnc"], 60), alpha = 0.4, color='steelblue')
    axB.fill_between(tAvg(methaneAvg, 60).index, tAvg(methaneAvg["transFlux"] + methaneAvg["transFluxUnc"], 60),
                     tAvg(methaneAvg["transFlux"] - methaneAvg["transFluxUnc"], 60), alpha = 0.4, color='orange')
    axB.legend()
    axB.set_xticks(axB.get_xticks(), axB.get_xticklabels(), rotation=45, ha='right')
    axB.set_title("Hourly Average Flux Rates", fontweight='bold')
    axB.set_xlabel("Time $[YYYY-DD-MM]$")
    axB.set_ylabel("Methane Flux $[\dfrac{g}{day}m^2]$")
    plt.show()

    figA.savefig(f"Figures/Flux_Rates_Boron{pcbNum}.png")
    figB.savefig(f"Figures/Hourly_Flux_Rates_Boron{pcbNum}.png")
    
except Exception as e:
    stopLoading(lE, lT, done=False)
    raise e
    
stopLoading(lE, lT)

""" ---- Archived Plots ----
#%% ARCHIVE
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
"""