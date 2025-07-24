"""
Averaged Setpoint Generator & Calibration Code
Sam Hill & Jessica Goff

 ----------------------------------------------------------------- DESCRIPTION -----------------------------------------------------------------
This code is used to generate averaged setpoint data from physical calibration. The averaged data will be input directly into the calibration code.

MODULE DEPENDENCIES (In "/Python" directory):
    "loadingUI.py"      —   Tools for an improved UI so that the user knows what the script is doing at all times! 
    "sheets_puller.py"  —   Tools to retrieve spreadsheet data via Google Sheets API.
    "credentials.json"  —   Essential Google API credentials for "sheets_puller.py"

INPUT:
    Setpoint file .xlsx, this file will preload with a defaultPath unless specified otherwise.
        
    USER INPUTS
        Start Time: Start of setpoint visualization
        End Time: End of setpoint visualization
        SS_start: Beginning of steady state in minutes
        SS_end: End of steady state in minutes
        sheet: sheet within the Google Sheet to draw data from
    
OUTPUT:
    Averaged Setpoint File named Averaged Setpoint+date.xlsx
"""

# Importing loading UI essentials...
from Python.loadingUI import startLoading, stopLoading
lE, lT = startLoading(message = "Importing libraries")

# Importing essential libraries...
try:
    from Python.sheets_puller import sheetPuller
    import os, pickle, textwrap
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tkinter as tk
    from tkinter import filedialog
    from datetime import timedelta, datetime as dt
    from sklearn import linear_model
    from sklearn.base import clone
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from scipy.stats import linregress
    from sklearn.linear_model import LinearRegression
    from pathlib import Path
    from matplotlib.backends.backend_pdf import PdfPages
except Exception as e:
    stopLoading(lE, lT, done=False)
    raise e
stopLoading(lE, lT)

# -------------------------------------------------------------- Calibration code begins on line 189 -----------------------------------------------------------

#%% ------------------------------------------------------------------ User-Defined Variables ------------------------------------------------------------------
# Importing setpoint file
filePrompt = False      # Prompt the user to select a specific setpoint file?
                        # If filePrompt is false, defaultPath will be used instead
defaultPath = "MetaData_060325.csv"

pcbNum = 4              # What Boron sensor are you using?

# Setting start & end times
startTime = pd.to_datetime("2025-06-03 15:23:00")      # Must be within selected dataset
endTime = pd.to_datetime("2025-06-09 16:17:00")        # Must be in [yyyy-mm-dd hh-mm-ss] format

# Setting steady state time range within start & end times
SS_start = 11       # [min]
SS_end   = 15       # [min]

#%% Importing data...
lE, lT = startLoading(message = "Importing data", t=0.25)

try: MOX = sheetPuller(f"Boron {pcbNum}", "06.03.25 Calibration")
except Exception as e:
    stopLoading(lE, lT, done=False)
    print("\nAn error was encountered while importing data!")
    raise e

# If user wants to be prompted to choose file
if filePrompt:
    print("Select Setpoints File in the Dialog Window to Continue!")
    root = tk.Tk().withdraw()
    setpoints_file = filedialog.askopenfile(title="Select Setpoints File", filetypes=(("CSV files", "*.csv"), ("all files", "*.*")))
    Setpoints = pd.read_csv(setpoints_file.name)
else:
    Setpoints = pd.read_csv(defaultPath)                        # defaultPath specified at top of code

stopLoading(lE, lT)

#%% Creating folders for figures, reports, coefficients, and box concentrations...
for path in ["Coefficients", "Figures", "Reports", "Box_Concentrations"]:
    Path(path).mkdir(parents=True, exist_ok=True)
    
#%% Cleaning MOX data & resampling...
print("Cleaning and preparing data...")

# Converting 'Time' column and coercing errors to NaT (Not a Time)
MOX['Time'] = pd.to_datetime(MOX['Time'], format='%Y-%m-%d %H:%M:%S', errors = 'coerce')

# Dropping rows with invalid dates and setting time as the index
MOX = MOX.set_index('Time')
MOX = MOX.sort_index()

# Removing rows with the boot-up timestamp
MOX = MOX[MOX.index != pd.Timestamp("1999-12-31 17:00:12")]

# Filtering to the date range of interest
print(f"\n---- MOX Timeframe Adjustments ----\nMOX size before filtering: {len(MOX)}")
# Checking if the date range does not exist
if not MOX.empty:
    try:
        MOX = MOX.loc[(MOX.index >= startTime) & (MOX.index <= endTime)]
    except KeyError:
        print("Warning: The specified startTime or endTime was not found in the MOX data. Skipping date range filter.")
print(f"MOX size after filtering: {len(MOX)}\n")

# Dropping irrelevant columns
MOX.drop(columns = [
    "Main EC_Worker (mV)", "Main EC_Aux (mV)", "Pressure (hPa)", "GasResistance (Ohms)", "FlowRate (slm)", "FlowTemp (C)"], errors = 'ignore', inplace = True)

# Renaming columns to remove units
MOX.rename(columns = {"BO TGS2600 (mV)" : "BO TGS2600", "BO TGS2602 (mV)" : "BO TGS2602", "BO TGS2611 (mV)" : "BO TGS2611", "BO EC_Worker (mV)" : "BO EC_Worker",
                      "BO EC_Aux (mV)" : "BO EC_Aux", "SGX_Analog (mV)" : "SGX_Analog", "SGX_Digital (ppm)" : "SGX_Digital", "BO Temperature (C)" : "BO Temperature",
                      "BO Pressure (hPa)" : "BO Pressure", "BO Humidity (%)" : "BO Humidity", "BO GasResistance (Ohms)" : "BO GasResistance", 
                      "Main TGS2600 (mV)" : "aTGS2600", "Main TGS2602 (mV)" : "aTGS2602", "Main TGS2611 (mV)" : "aTGS2611", "Temperature (C)" : "aTemp", 
                      "Humidity (%)" : "aHumidity"}, inplace = True)

# Cleaning data values and resampling for plotting
MOX = (MOX.mask(MOX == 404, np.nan).infer_objects(copy = False)
       .dropna().resample("60s").last().dropna())

#%% Cleaning Setpoints data...
Setpoints = Setpoints.rename(columns = {'Date [mm/dd] Start Time [hh:mm:ss]': 'time'})  # Renaming time column

# Converting 'time' column to datetime and coercing errors
Setpoints['time'] = pd.to_datetime(Setpoints['time'], format="mixed")  
Setpoints = Setpoints.set_index('time')                         # Dropping nan rows and setting time as index

# Isolating start and end times in Setpoints
Setpoints = Setpoints[(Setpoints.index > startTime) & (Setpoints.index < endTime)]

#%% Generating setpointavg dataframe...
print("Creating setpoint avg dataframe...\n")

results = [] # A list to hold our complete, valid rows.

# Using .iterrows() to get both the timestamp and the row data from Setpoints
for timestamp, row in Setpoints.iterrows():
    try:
        window_data = MOX.loc[timestamp + timedelta(minutes=SS_start): timestamp + timedelta(minutes=SS_end)]

        # Calculates mean of sensor data if it exists
        if not window_data.empty:
            sensor_means = window_data.mean()
            
            # Creating a dictionary from the sensor means
            new_row_dict = sensor_means.to_dict()
            
            # Adding the corresponding H2S and CH4 values from the current Setpoints row
            new_row_dict['H2S'] = row['C_H2S [ppm]']
            new_row_dict['Setpoint'] = row['C_CH4 [ppm]']
            
            # Appending the complete dictionary for this valid row to our list
            results.append(new_row_dict)
            
        else:
            # If the window is empty, print message with timestamp for missing data but do not include 0s in dataframe
            print(f"No data found for window around {timestamp.strftime('%Y-%m-%d %H:%M:%S')}. Skipping.")
            pass

    except Exception as e:
        print(f"An unexpected error occurred for timestamp {timestamp}: {e}")           # In case of other errors in data

# Creating the final dataframe from our list of processed dictionaries
setpointavg = pd.DataFrame(results)

# Sorting and resetting the index based on setpoints
setpointavg = (
    setpointavg.sort_values(by = "Setpoint", ascending=True)
               .reset_index(drop=True))

#%% Creating average setpoint file...
setpointavg.to_excel(f"Averaged_Setpoints_boron{pcbNum}.xlsx")
print("\nComplete! Averaged setpoints file created :D")

lE, lT = startLoading(message="Performing calibration..")

#%% ------------------------------------------------------------------ CALIBRATION CODE ------------------------------------------------------------------
"""
 ----------------------------------------------------------------- DESCRIPTION -----------------------------------------------------------------
This code utilizes the averaged setpoint data and generates a calibration regression to map the sensors' methane response.

INPUTS:
    setpointavg dataframe from Get_Setpoints.
    
OUTPUTS:
    PDF report with plots, RMSE, R² values, and regression equations.
    Plots and file for residuals.
    4 regression models — reglow.plk, regmid.plk, reghigh.plk, regamb.pkl — found in /Coeffiecients.
"""

#%% Data Wrangling, split into training and delivering ranges...
try:
    # Creating feature interactions and ratios
    setpointavg["Temp*Humid"] = setpointavg["BO Temperature"] * setpointavg["BO Humidity"]
    setpointavg["R11_00"] = setpointavg["BO TGS2611"] / setpointavg["BO TGS2600"]
    setpointavg["aR11_00"] = setpointavg["aTGS2611"] / setpointavg["aTGS2600"]
    setpointavg["2611*Temp"] = setpointavg["BO TGS2611"] * setpointavg["BO Temperature"]
    setpointavg["SGX*Temp"] = setpointavg["SGX_Digital"] * setpointavg["BO Temperature"]
    
    # Training ranges from the TRAINING dataset
    ambTrain = setpointavg[setpointavg["Setpoint"] <= 30]
    lowTrain = setpointavg[(setpointavg["Setpoint"] <= 71) & (setpointavg["Setpoint"] >= 0)]
    medTrain = setpointavg[(setpointavg["Setpoint"] >= 60) & (setpointavg["Setpoint"] <= 1000)]
    highTrain = setpointavg[setpointavg["Setpoint"] >= 600]
        
    # Delivering (testing) ranges from the TESTING dataset
    ambTest = setpointavg[setpointavg["Setpoint"] <= 30]
    lowTest = setpointavg[(setpointavg["Setpoint"] <= 71) & (setpointavg["Setpoint"] >= 0)]
    medTest = setpointavg[(setpointavg["Setpoint"] > 71) & (setpointavg["Setpoint"] <= 750)]
    highTest = setpointavg[setpointavg["Setpoint"] > 750]
    
    #%% Generating regression models for each range...
    # Creating base regression models
    # Polynomial degree 2
    polynomial_model_deg2 = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('linear', linear_model.LinearRegression())
    ])
    
    # Polynomial degree 3
    polynomial_model_deg3 = Pipeline([
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('linear', linear_model.LinearRegression())
    ])
    
    # --- Model Assignments ---
    low_reg = clone(polynomial_model_deg3)
    med_reg = clone(polynomial_model_deg3)
    high_reg = clone(polynomial_model_deg2)
    amb_reg = clone(polynomial_model_deg2)
    
    # Define the feature sets with the original column names
    Xamb = ambTrain[["aTGS2600", "aTGS2611", "aTGS2602", "aTemp", "aHumidity", "aR11_00"]].copy()
    Xlow = lowTrain[["BO TGS2600", "BO TGS2611", "Temp*Humid", "R11_00"]].copy()
    Xmed = medTrain[["BO TGS2600", "BO TGS2611", "2611*Temp", "Temp*Humid", "BO Humidity"]].copy()
    Xhigh = highTrain[["BO TGS2611", "SGX_Digital", "BO Temperature", "BO Humidity", "2611*Temp", "SGX*Temp"]].copy()
    
    xAmb = ambTest[["aTGS2600", "aTGS2611", "aTGS2602", "aTemp", "aHumidity", "aR11_00"]].copy()
    Xlo = lowTest[["BO TGS2600", "BO TGS2611", "Temp*Humid", "R11_00"]].copy()
    Xmid = medTest[["BO TGS2600", "BO TGS2611", "2611*Temp", "Temp*Humid", "BO Humidity"]].copy()
    Xhi = highTest[["BO TGS2611", "SGX_Digital", "BO Temperature", "BO Humidity", "2611*Temp", "SGX*Temp"]].copy()
    
    # Renaming the columns on the DataFrames before fitting the model.
    rename_dict = {
        "BO TGS2600": "TGS2600", "BO TGS2611": "TGS2611", "BO TGS2602": "TGS2602", "BO Temperature": "Temp",
        "BO Humidity": "Humidity"
    }
    
    Xlow.rename(columns=rename_dict, inplace=True)
    Xmed.rename(columns=rename_dict, inplace=True)
    Xhigh.rename(columns=rename_dict, inplace=True)
    Xlo.rename(columns=rename_dict, inplace=True)
    Xmid.rename(columns=rename_dict, inplace=True)
    Xhi.rename(columns=rename_dict, inplace=True)
    
    # Fit the regression models using the DataFrames with the NEW, clean names
    amb_reg.fit(Xamb, ambTrain["Setpoint"])
    low_reg.fit(Xlow, lowTrain["Setpoint"])
    med_reg.fit(Xmed, medTrain["Setpoint"])
    high_reg.fit(Xhigh, highTrain["Setpoint"])
    
    # Creating a dataframe with the actual and predicted values from models
    reg_model_diff_amb = pd.DataFrame({"Actual value": ambTest["Setpoint"], "Predicted value": amb_reg.predict(xAmb)})
    reg_model_diff_low = pd.DataFrame({"Actual value": lowTest["Setpoint"], "Predicted value": low_reg.predict(Xlo)})
    reg_model_diff_med = pd.DataFrame({'Actual value': medTest["Setpoint"], "Predicted value": med_reg.predict(Xmid)})
    reg_model_diff_high = pd.DataFrame({"Actual value": highTest["Setpoint"], 'Predicted value': high_reg.predict(Xhi)})
    
    # Combining linear regressions to create piecewise regression
    Combined = pd.concat([reg_model_diff_low, reg_model_diff_med, reg_model_diff_high])
    
    # Dropping duplicates
    Combined = (Combined.reset_index().drop_duplicates(subset = 'index', keep = 'last').set_index('index').sort_index())
    
    # Calculating RMS for entire regression, low, medium, & high RMS
    RMS = np.sqrt(mean_squared_error(Combined["Actual value"], Combined["Predicted value"]))
    RMSamb = np.sqrt(mean_squared_error(reg_model_diff_amb["Actual value"], reg_model_diff_amb["Predicted value"]))
    RMSlow = np.sqrt(mean_squared_error(reg_model_diff_low["Actual value"], reg_model_diff_low["Predicted value"]))
    RMSmid = np.sqrt(mean_squared_error(reg_model_diff_med["Actual value"], reg_model_diff_med["Predicted value"]))
    RMShigh = np.sqrt(mean_squared_error(reg_model_diff_high["Actual value"], reg_model_diff_high["Predicted value"]))
    
    # Calculating R² for each model
    r2_amb = r2_score(reg_model_diff_amb["Actual value"], reg_model_diff_amb["Predicted value"])
    r2_low = r2_score(reg_model_diff_low["Actual value"], reg_model_diff_low["Predicted value"])
    r2_med = r2_score(reg_model_diff_med["Actual value"], reg_model_diff_med["Predicted value"])
    r2_high = r2_score(reg_model_diff_high["Actual value"], reg_model_diff_high["Predicted value"])
    r2_combined = r2_score(Combined["Actual value"], Combined["Predicted value"])
    
    # Printing R² and RMS values for each model
    print("\n\n------ R² & RMS ERROR REPORT ------")
    print(f"Ambient Regression Model (Poly Deg 2) --> RMS: {RMSamb:.4f} [ppm], R²: {r2_amb:.4f}")
    print(f"Low Range Model (Poly Deg 3) --> RMS: {RMSlow:.4f} [ppm], R²: {r2_low:.4f}")
    print(f"Medium Range Model (Poly Deg 3) -> RMS: {RMSmid:.4f} [ppm], R²: {r2_med:.4f}")
    print(f"High Range Model (Poly Deg 2)--> RMS: {RMShigh:.4f} [ppm], R²: {r2_high:.4f}")
    print(f"Combined Model --------------> RMS: {RMS:.4f} [ppm], R²: {r2_combined:.4f}\n")
    
    #%% Regression equations...
    # Accessing PolynomialFeatures steps from each pipeline
    low_poly = low_reg.named_steps['poly']
    med_poly = med_reg.named_steps['poly']
    high_poly = high_reg.named_steps['poly']
    
    # Formatting equations for readability
    def format_equation(coefs, features, intercept, threshold=0.00001, sig_figs=4):
        terms = []
        for coef, feat in zip(coefs, features):
            if abs(coef) < threshold: continue
            coef_str = f"{abs(coef):.{sig_figs}g}"
            cleaned_feat = feat.replace(' ', ' * ')
            sign = "-" if coef < 0 else "+"
            terms.append(f" {sign} {coef_str} * {cleaned_feat}")
    
        intercept_sign = "-" if intercept < 0 else "+"
        intercept_str = f" {intercept_sign} {abs(intercept):.{sig_figs}g}"
    
        if not terms: return f"Setpoint = {intercept:.{sig_figs}g}"
    
        full_equation = "".join(terms).lstrip(" +")
        full_equation += intercept_str
        if full_equation.startswith('-'): full_equation = "- " + full_equation[1:]
        return f"Concentration = {full_equation.strip()}"
    
    print("\n\n------ REGRESSION EQUATIONS ------")
    
    # The calls are now simple, with no extra arguments needed.
    lowEQ = format_equation(
        low_reg.named_steps['linear'].coef_,
        low_poly.get_feature_names_out(),
        low_reg.named_steps['linear'].intercept_
    )
    print("\nLow Regression Equation:\n", lowEQ)
    
    medEQ = format_equation(
        med_reg.named_steps['linear'].coef_,
        med_poly.get_feature_names_out(),
        med_reg.named_steps['linear'].intercept_
    )
    print("\nMedium Regression Equation:\n", medEQ)
    
    highEQ = format_equation(
        high_reg.named_steps['linear'].coef_,
        high_poly.get_feature_names_out(),
        high_reg.named_steps['linear'].intercept_
    )
    print("\nHigh Regression Equation:\n", highEQ)

except Exception as e:
    stopLoading(lE, lT, done=False)
    raise e

stopLoading(lE, lT)

#%% Saving regressions...
# Loop to save the pickle files
for name, data_object in [('lowreg', low_reg), ('midreg', med_reg), ('highreg', high_reg), ('ambreg', amb_reg)]:
    filepath = os.path.join("Coefficients", f"{name}_Boron{pcbNum}.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(data_object, f)

# Loop to save the text files
for name, value in [('low', RMSlow), ('med', RMSmid), ('high', RMShigh), ('amb', RMSamb)]:
    filepath = os.path.join("Coefficients", f"RMSE_{name}_Boron{pcbNum}.txt")
    with open(filepath, "w") as f:
        f.write(str(value))

#%% Plotting sensor response against setpoints...
lE, lT = startLoading(message = "Plotting")

try:
    fig1,ax1 = plt.subplots(1, 1, figsize = (9,5), dpi = 600)      # Creating figure
    ax1.plot(MOX.index[(MOX.index > startTime) & (MOX.index < endTime)],
             MOX["BO TGS2602"][(MOX.index > startTime) & (MOX.index < endTime)], label = "TGS02")     # Plotting TGS2602 sensor
    ax1.plot(MOX.index[(MOX.index > startTime) & (MOX.index < endTime)],
             MOX["BO TGS2611"][(MOX.index > startTime) & (MOX.index < endTime)], label = "TGS11")     # Plotting TGS2611 sensor
    ax1.plot(MOX.index[(MOX.index > startTime) & (MOX.index < endTime)],
             MOX["BO TGS2600"][(MOX.index > startTime) & (MOX.index < endTime)], label = "TGS00")     # Plotting TGS2600 sensor
    ax1.plot(MOX.index[(MOX.index > startTime) & (MOX.index < endTime)],
             MOX["SGX_Digital"][(MOX.index > startTime) & (MOX.index < endTime)], label = "SGX")     # Plotting SGX sensor
    # Plotting vertical lines for each setpoint time across the steady state period
    for i in range(len(Setpoints)):
        ax1.axvline(x = Setpoints.index[i], color = "black", linewidth = 0.5, zorder=-1, alpha=0.5)
        ax1.axvspan(Setpoints.index[i] + timedelta(minutes = SS_start),
                    Setpoints.index[i] + timedelta(minutes = SS_end), alpha = 0.5, zorder=-1)
    # Creating legend and labelling axes
    ax1.legend()
    ax1.set_xlabel("Time $[MM-DD-HH]$", fontsize=12)
    ax1.set_ylabel("TGS2611 Response $[mV]$", fontsize=12)
    ax1.set_xlim([startTime, endTime])        # Plotting between the start & end times
    ax1.set_title(f"Sensor Readings on {dt.date(startTime)}", fontsize=18, fontweight='bold', pad=15)
    plt.show()
    
    #%% Plotting SGX signal...
    fig2,ax2 = plt.subplots(1, 1, figsize = (9,5), dpi = 600)       # Creating figure
    ax2.plot(MOX.index[(MOX.index > startTime) & (MOX.index < endTime)],
             MOX["SGX_Digital"][(MOX.index > startTime) & (MOX.index < endTime)],label = "SGX")      # Plotting SGX sensor
    # Labelling axes
    ax2.set_xlabel("Time $[YYYY-DD-MM]$", fontsize=12)
    ax2.set_ylabel("SGX Signal $[ppm]$", fontsize=12)
    ax2.set_xlim([startTime, endTime])                # Plotting between the start & end times
    ax2.grid(True, zorder=-1, color="grey", alpha=0.5)
    ax2.set_title(f"SGX Signal Beginning on {dt.date(startTime)}", fontsize=18, fontweight='bold', pad=15)
    ax2.tick_params(axis='x', labelrotation=45)        # Rotating x-axis labels for readability
    plt.show()
    
    #%% Plotting sensor accuracy across temperatures...
    def sensAccuracy():
        
        def colorClasses(temps):
            masks = [(pd.qcut(temps, q=4, labels=[0,1,2,3]) == i).values for i in range(4)]
            bins = pd.qcut(temps, q=4, retbins=True)[1]
            names = [f"Below {bins[1]:.1f} $[°C]$", f"Between {bins[1]:.1f} & {bins[2]:.1f} $[°C]$",
                     f"Between {bins[2]:.1f} & {bins[3]:.1f} $[°C]$", f"Over {bins[3]:.1f} $[°C]$"]
            return zip(["midnightblue", "blue", "cornflowerblue", "lightblue"], masks, names)
    
        Combined["temperature"] = setpointavg["BO Temperature"]
        
        fig, ax = plt.subplots(1, 1, dpi=600, figsize=(10,7))
        
        setpoint_anchors = np.unique(Combined["Actual value"].values)
        # Create a perfectly linear sequence that corresponds to our anchors.
        linear_positions = np.linspace(0, 1, len(setpoint_anchors))
        
        min_val = setpoint_anchors.min()
        max_val = setpoint_anchors.max()
        
        # Define the padding amount (e.g., 5% of the core data range)
        padding_amount = (max_val - min_val) * 0.025
        
        # Calculating the final, padded boundaries for our scale
        final_min_limit = min_val - padding_amount
        final_max_limit = max_val + padding_amount
        
        # Creating a new, wider set of anchors that includes the padded limits
        scale_anchors = np.concatenate(([final_min_limit], setpoint_anchors, [final_max_limit]))
        scale_anchors = np.unique(scale_anchors) # Ensure no duplicates if padding is zero
        
        # The linear positions now correspond to these new padded anchors
        linear_positions = np.linspace(0, 1, len(scale_anchors))
        
        # Defining the SHAPE-AWARE transformation functions based on the padded anchors
        def final_forward_transform(values):
            """Maps data values to the padded linear scale."""
            values = np.asarray(values)
            original_shape = values.shape
            flattened_values = values.ravel()
            interpolated_ranks = np.interp(flattened_values, scale_anchors, linear_positions)
            return interpolated_ranks.reshape(original_shape)
        
        def final_inverse_transform(ranks_input):
            """Maps linear positions back to data values from the padded scale."""
            ranks_input = np.asarray(ranks_input)
            original_shape = ranks_input.shape
            flattened_ranks = ranks_input.ravel()
            interpolated_values = np.interp(flattened_ranks, linear_positions, scale_anchors)
            return interpolated_values.reshape(original_shape)
        
        # Applying the SAME universal padded scale to BOTH axes.
        ax.set_xscale('function', functions=(final_forward_transform, final_inverse_transform))
        ax.set_yscale('function', functions=(final_forward_transform, final_inverse_transform))
        
        # A straight line is now a true 1:1 reference on our new universal scale
        # We plot it from the padded min to padded max to span the whole view
        ax.plot([final_min_limit, final_max_limit], [final_min_limit, final_max_limit],
                color='black', linestyle='--', alpha=0.7, label='1-to-1 Line')
        
        for color, mask, label in colorClasses(Combined["temperature"]):
            ax.scatter(Combined["Actual value"][mask], Combined["Predicted value"][mask], edgecolor = color,
                       label=label, marker='o', facecolors='none', linewidths=1.5)
        
        # Finalize the plot
        ax.legend()
        ax.set_xlabel("Actual Setpoint Value $[ppm]$", fontsize=12)
        ax.set_ylabel("Predicted Value $[ppm]$ (Scaled Against Setpoints)", fontsize=12)
        fig.suptitle("One-to-One Calibration Plot", fontsize=15, fontweight='bold')
        ax.grid(True, zorder=-1, color="grey", alpha=0.5)
        
        # Set the ticks on both axes to be the original setpoint anchors
        ax.set_xticks(setpoint_anchors)
        ax.set_yticks(setpoint_anchors)
            
        ax.legend()
        plt.show()
        return fig, ax
    
    fig3, ax3 = sensAccuracy()
    
    #%% Plotting actual vs predicted values...
    def modelAccuracy():
        fig, ax = plt.subplots(1, 1, dpi=600, figsize=(10,7))
        
        setpoint_anchors = np.unique(Combined["Actual value"].values)
        # Create a perfectly linear sequence that corresponds to our anchors.
        linear_positions = np.linspace(0, 1, len(setpoint_anchors))
        
        min_val = setpoint_anchors.min()
        max_val = setpoint_anchors.max()
        
        # Define the padding amount (e.g., 5% of the core data range)
        padding_amount = (max_val - min_val) * 0.05
        
        # Calculate the final, padded boundaries for our scale
        final_min_limit = min_val - padding_amount
        final_max_limit = max_val + padding_amount
           
        # Create a new, wider set of anchors that includes the padded limits
        scale_anchors = np.concatenate(([final_min_limit], setpoint_anchors, [final_max_limit]))
        scale_anchors = np.unique(scale_anchors) # Ensure no duplicates if padding is zero
        
        # The linear positions now correspond to these new padded anchors
        linear_positions = np.linspace(0, 1, len(scale_anchors))
        
        # Defining the SHAPE-AWARE transformation functions based on the padded anchors
        def final_forward_transform(values):
            """Maps data values to the padded linear scale."""
            values = np.asarray(values)
            original_shape = values.shape
            flattened_values = values.ravel()
            interpolated_ranks = np.interp(flattened_values, scale_anchors, linear_positions)
            return interpolated_ranks.reshape(original_shape)
        
        def final_inverse_transform(ranks_input):
            """Maps linear positions back to data values from the padded scale."""
            ranks_input = np.asarray(ranks_input)
            original_shape = ranks_input.shape
            flattened_ranks = ranks_input.ravel()
            interpolated_values = np.interp(flattened_ranks, linear_positions, scale_anchors)
            return interpolated_values.reshape(original_shape)
        
        # Applying the universal padded scale to y-axis
        ax.set_yscale('function', functions=(final_forward_transform, final_inverse_transform))
        
        # Plotting data
        ax.plot(reg_model_diff_low.index, reg_model_diff_low["Predicted value"], label = 'Low Range Prediction', color = "mediumseagreen")   # Plotting low range 
        ax.fill_between(reg_model_diff_low.index, reg_model_diff_low["Predicted value"] - RMSlow,
                         reg_model_diff_low["Predicted value"] + RMSlow,color = 'mediumseagreen', alpha = 0.25, zorder=-1)     # Filling in 
        ax.plot(reg_model_diff_med.index,reg_model_diff_med["Predicted value"], label = 'Medium Range Prediction', color = "orchid")         # Plotting mid range
        ax.fill_between(reg_model_diff_med.index, reg_model_diff_med["Predicted value"] - RMSmid,
                         reg_model_diff_med["Predicted value"] + RMSmid,color = 'orchid', alpha = 0.25, zorder=-1)
        ax.plot(reg_model_diff_high.index, reg_model_diff_high["Predicted value"], label = 'High Range Prediction', color = "orange")        # Plotting high range
        ax.fill_between(reg_model_diff_high.index, reg_model_diff_high["Predicted value"] - RMShigh,
                         reg_model_diff_high["Predicted value"] + RMShigh,color = 'orange', alpha = 0.25, zorder=-1)
        ax.plot(Combined["Actual value"], color = 'k', label = 'Setpoints', linestyle = '--')
        
        # Finalizing the plot
        ax.legend()
        ax.set_xlabel("Setpoint Index", fontsize=12)
        ax.set_ylabel("Methane Concentration $[ppm]$ (Scaled Against Setpoints)", fontsize=12)
        fig.suptitle("Low, Medium, and High Regression Response to Setpoints", fontsize=15, fontweight='bold')
        ax.grid(True, zorder=-1, color="grey", alpha=0.5)
        
        # Set the ticks on both axes to be the original setpoint anchors
        #ax.set_xticks(setpoint_anchors)
        ax.set_yticks(setpoint_anchors)
        
        ax.legend()
        plt.show()
        return fig, ax
    
    fig4, ax4 = modelAccuracy()
    
    #%% Creating plot with regression equations, RMSE, and R² values
    fig5, ax5 = plt.subplots(dpi=600, figsize = (5, 7))
    
    # Turning off axes
    ax5.axis("off")
    
    # Creating header
    ax5.text(0.5, 0.95, "------ R² & RMS Error Report ------",
             fontsize=15, ha='center', transform=ax5.transAxes)
    
    # RMS & R² stats
    ax5.text(0.5, 0.89,
             f"""Low Range Model ---> RMSE: {RMSlow:.4f} [ppm], R²: {r2_low:.4f}
    Medium Range Model ---> RMSE: {RMSmid:.4f} [ppm], R²: {r2_med:.4f}
    High Range Model ---> RMSE: {RMShigh:.4f} [ppm], R²: {r2_high:.4f}
    Combined Model ---> RMSE: {RMS:.4f} [ppm], R²: {r2_combined:.4f}""",
             fontsize=12, ha='center', va='top', transform=ax5.transAxes,
             fontfamily='monospace')
    
    # Regression equation header
    ax5.text(0.5, 0.65, "------ Regression Equations ------",
             fontsize=15, ha='center', transform=ax5.transAxes)
    
    # Define a helper function to wrap and draw long text
    def draw_wrapped_text(ax, text, y_pos, label, font_size=8, max_width=120):
        wrapped = textwrap.wrap(text, width=max_width)
        full_text = f"{label}\n" + "\n".join(wrapped)
        ax.text(0.5, y_pos, full_text, ha='center', va='top',
                fontsize=font_size, transform=ax.transAxes,
                fontfamily='monospace')
    
    # Add each equation
    draw_wrapped_text(ax5, lowEQ, 0.62, "Low EQ:", font_size=8)
    draw_wrapped_text(ax5, medEQ, 0.42, "Medium EQ:", font_size=8)
    draw_wrapped_text(ax5, highEQ, 0.1, "High EQ:", font_size=8)
    
    #%% Recreating correlogram with pairgrid...
    # Choosing which MOX features to remove from the correlogram
    setpointavg.drop(columns = ["BO EC_Worker", "BO EC_Aux", "SGX_Analog", "SGX_Digital", "BO GasResistance", "H2S"], errors = 'ignore', inplace = True)
    
    # Function to scatter with regression and R²
    def scatter_with_r2(x, y, **kwargs):
        ax = plt.gca()
        ax.scatter(x, y, alpha=0.6)
    
        # Fitting linear regression
        if len(np.unique(x)) > 1:
            slope, intercept, r_value, *_ = linregress(x, y)
            ax.plot(x, slope * x + intercept, color="red", alpha=0.7)
            ax.text(0.05, 0.9, f"$R^2$ = {r_value**2:.2f}", transform=ax.transAxes, fontsize=9)
    
    # Creating full grid
    fig6 = sns.PairGrid(setpointavg, diag_sharey=False)
    
    # Plotting lower and upper triangles
    fig6.map_lower(scatter_with_r2)
    fig6.map_upper(scatter_with_r2)
    
    # Plotting diagonal histograms
    fig6.map_diag(sns.histplot, kde=False, color="skyblue")
    
    plt.tight_layout()
    plt.show()
    
    #%% Temperature vs Setpoints...
    fig7, ax7 = plt.subplots(1,1, dpi=600, figsize=(10,5))
    ax7.scatter(Setpoints.index, Setpoints["T [C]"], label="Setpoint Values")
    ax7.scatter(MOX.index, MOX["BO Temperature"], label="Chamber Values")
    ax7.set_xlabel("Time")
    ax7.set_ylabel("Temperature $[C]$")
    ax7.legend()
    
    #%% Residuals plot...
    # Consolidate model results and calculate residuals
    low_res_df = pd.DataFrame({
        'True Value (ppm)': reg_model_diff_low['Actual value'],
        'Residual Error (ppm)': reg_model_diff_low['Actual value'] - reg_model_diff_low['Predicted value'],
        'Range': 'Low'
    })
    
    med_res_df = pd.DataFrame({
        'True Value (ppm)': reg_model_diff_med['Actual value'],
        'Residual Error (ppm)': reg_model_diff_med['Actual value'] - reg_model_diff_med['Predicted value'],
        'Range': 'Medium'
    })
    
    high_res_df = pd.DataFrame({
        'True Value (ppm)': reg_model_diff_high['Actual value'],
        'Residual Error (ppm)': reg_model_diff_high['Actual value'] - reg_model_diff_high['Predicted value'],
        'Range': 'High'
    })
    
    # Combine all ranges into a single dataframe for plotting
    residuals_df = pd.concat([low_res_df, med_res_df, high_res_df]).reset_index(drop=True)
    
    # Define and apply the custom x-axis scale with PADDING 
    if not residuals_df.empty:
        # Get the core data points that will serve as tick marks
        setpoint_anchors = np.unique(residuals_df["True Value (ppm)"].values)
    
        # Calculate the padded data range for the scale's transformation
        min_val = setpoint_anchors.min()
        max_val = setpoint_anchors.max()
        padding_amount = (max_val - min_val) * 0.05
    
        # Create the final anchors for the transformation, including the padding
        scale_anchors = np.unique(np.concatenate(([min_val - padding_amount], setpoint_anchors, [max_val + padding_amount])))
        
        # Create the linear positions that correspond to our new padded anchors
        linear_positions = np.linspace(0, 1, len(scale_anchors))
    
        # Define the transformation functions based on the full padded scale
        def forward_transform(values):
            """Maps data values to the padded linear scale."""
            return np.interp(values, scale_anchors, linear_positions)
    
        def inverse_transform(ranks):
            """Maps linear positions back to data values from the padded scale."""
            return np.interp(ranks, linear_positions, scale_anchors)
    
        # Generate the final plot 
        fig_res, ax_res = plt.subplots(figsize=(11, 7), dpi=150)
        ax_res.set_xscale('function', functions=(forward_transform, inverse_transform))
    
        # Create the scatter plot using the consolidated residuals data
        sns.scatterplot(
            data=residuals_df,
            x="True Value (ppm)",
            y="Residual Error (ppm)",
            hue="Range", 
            style="Range", 
            s=70, 
            ax=ax_res, 
            palette="viridis"
        )
    
        # Add a horizontal line at y=0 for a zero-error reference
        ax_res.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error Line')
    
        # Customize the plot with titles and labels
        ax_res.set_title("Residuals Plot (Trained on Full Dataset)", fontsize=18, pad=20)
        ax_res.set_xlabel("True CH4 Concentration (Setpoint) [ppm] - Custom Scale", fontsize=12)
        ax_res.set_ylabel("Residual Error (True - Predicted) [ppm]", fontsize=12)
        ax_res.grid(True, which='both', linestyle='--', linewidth=0.5)
    
        # Set the x-axis ticks to be the original setpoint values for clarity
        ax_res.set_xticks(setpoint_anchors)
        
        # Reorder the legend to be more intuitive
        handles, labels = ax_res.get_legend_handles_labels()
        try:
            zero_line_idx = labels.index('Zero Error Line')
            handles.append(handles.pop(zero_line_idx))
            labels.append(labels.pop(zero_line_idx))
        except ValueError:
            pass # In case the label isn't found
        ax_res.legend(handles=handles, labels=labels, title='Calibration Range')
    
        plt.tight_layout()
        plt.show()
    else:
        print("Could not generate residuals plot because no data was available.")
        
    #%% Residual Error vs. Temperature Plot
    # Consolidate results, calculate residuals, and add Temperature 
    low_res_df = pd.DataFrame({
        'True Value (ppm)': reg_model_diff_low['Actual value'],
        'Residual Error (ppm)': reg_model_diff_low['Actual value'] - reg_model_diff_low['Predicted value'],
        'Range': 'Low',
        'Temperature': lowTest['BO Temperature']
    })
    
    med_res_df = pd.DataFrame({
        'True Value (ppm)': reg_model_diff_med['Actual value'],
        'Residual Error (ppm)': reg_model_diff_med['Actual value'] - reg_model_diff_med['Predicted value'],
        'Range': 'Medium',
        'Temperature': medTest['BO Temperature']
    })
    
    high_res_df = pd.DataFrame({
        'True Value (ppm)': reg_model_diff_high['Actual value'],
        'Residual Error (ppm)': reg_model_diff_high['Actual value'] - reg_model_diff_high['Predicted value'],
        'Range': 'High',
        'Temperature': highTest['BO Temperature']
    })
    
    # Combine all ranges into a single dataframe for plotting
    residuals_vs_temp_df = pd.concat([low_res_df, med_res_df, high_res_df]).reset_index(drop=True)
    
    # Generate the plot
    if not residuals_vs_temp_df.empty:
        fig_temp_res, ax_temp_res = plt.subplots(figsize=(11, 7), dpi=150)
    
        # Create the scatter plot with Temperature on the x-axis
        sns.scatterplot(
            data=residuals_vs_temp_df,
            x="Temperature",
            y="Residual Error (ppm)",
            hue="Range", 
            style="Range", 
            s=70, 
            ax=ax_temp_res, 
            palette="coolwarm" # A good palette for temperature-related plots
        )
    
        # Add a horizontal line at y=0 for a zero-error reference
        ax_temp_res.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Zero Error Line')
    
        # Customize the plot with new titles and labels
        ax_temp_res.set_title("Methane Residual Error vs. Temperature", fontsize=18, pad=20)
        ax_temp_res.set_xlabel("BO Temperature (°C)", fontsize=12)
        ax_temp_res.set_ylabel("Residual Error (True - Predicted) [ppm]", fontsize=12)
        ax_temp_res.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Create the legend
        ax_temp_res.legend(title='Setpoint Range')
    
        plt.tight_layout()
        plt.show()
    else:
        print("Could not generate residuals vs. temperature plot because no data was available.")
        
    #%% Residual Error vs. Humidity Plot
    # Consolidate results, calculate residuals, and add Humidity
    low_res_df = pd.DataFrame({
        'True Value (ppm)': reg_model_diff_low['Actual value'],
        'Residual Error (ppm)': reg_model_diff_low['Actual value'] - reg_model_diff_low['Predicted value'],
        'Range': 'Low',
        'Humidity': lowTest['BO Humidity']
    })
    
    med_res_df = pd.DataFrame({
        'True Value (ppm)': reg_model_diff_med['Actual value'],
        'Residual Error (ppm)': reg_model_diff_med['Actual value'] - reg_model_diff_med['Predicted value'],
        'Range': 'Medium',
        'Humidity': medTest['BO Humidity']
    })
    
    high_res_df = pd.DataFrame({
        'True Value (ppm)': reg_model_diff_high['Actual value'],
        'Residual Error (ppm)': reg_model_diff_high['Actual value'] - reg_model_diff_high['Predicted value'],
        'Range': 'High',
        'Humidity': highTest['BO Humidity']
    })
    
    # Combine all ranges into a single dataframe for plotting
    residuals_vs_humid_df = pd.concat([low_res_df, med_res_df, high_res_df]).reset_index(drop=True)
    
    # Generate the plot
    if not residuals_vs_humid_df.empty:
        # Create new figure and axes objects for this plot
        fig_humid_res, ax_humid_res = plt.subplots(figsize=(11, 7), dpi=150)
    
        # Create the scatter plot with Humidity on the x-axis
        sns.scatterplot(
            data=residuals_vs_humid_df,
            x="Humidity",
            y="Residual Error (ppm)",
            hue="Range", 
            style="Range", 
            s=70, 
            ax=ax_humid_res, 
            palette="viridis" # A nice, general-purpose color palette
        )
    
        # Add a horizontal line at y=0 for a zero-error reference
        ax_humid_res.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Zero Error Line')
    
        # Customize the plot with new titles and labels
        ax_humid_res.set_title("Methane Residual Error vs. Humidity", fontsize=18, pad=20)
        ax_humid_res.set_xlabel("BO Humidity (%)", fontsize=12)
        ax_humid_res.set_ylabel("Residual Error (True - Predicted) [ppm]", fontsize=12)
        ax_humid_res.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Create the legend
        ax_humid_res.legend(title='Setpoint Range')
    
        plt.tight_layout()
        plt.show()
    else:
        print("Could not generate residuals vs. humidity plot because no data was available.")
    
except Exception as e:
    stopLoading(lE, lT, done=True)
    print("\n An error was encountered while generating figures!")
    raise e
    
stopLoading(lE, lT)

#%% Saving figures...
lE, lT = startLoading(message = "Exporting figures and data")

try:
    fig1.savefig(f"Figures/boron{pcbNum}Sensors.png", bbox_inches = "tight")
    fig2.savefig(f"Figures/boron{pcbNum}SGX.png", bbox_inches = "tight")
    fig3.savefig(f"Figures/boron{pcbNum}LinearRegression1to1.png", bbox_inches = "tight")
    fig4.savefig(f"Figures/boron{pcbNum}LinearRegression.png", bbox_inches = "tight")
    fig5.savefig(f"Figures/boron{pcbNum}RMSE&R^2.png", bbox_inches = "tight")
    fig6.savefig(f"Figures/boron{pcbNum}Correlogram.png", bbox_inches = "tight", dpi = 600)
    
    #%% Exporting figures to pdf report...
    def exportReport(fileName, *figs, dpi=600):
        print("\nGenerating PDF Report")
        with PdfPages(fileName) as pdf:
            for fig in figs:
                pdf.savefig(fig, bbox_inches='tight', pad_inches = 0.3, dpi=dpi)
        
    exportReport(f"Reports/boron{pcbNum}Cal.pdf", fig1, fig2, fig3, fig4, fig5)
    
    #%% Generating Residuals file...
    Combined["humidities"] = setpointavg["BO Humidity"]
    Combined["temperature"] = setpointavg["BO Temperature"]
    #Combined["pressure"] = setpointavg["B_Pressure"]
    #Combined["H2S"] = setpointavg["H2S"]
    Combined.to_csv(f"boron{pcbNum}Residuals.csv")
    
except Exception as e:
    stopLoading(lE, lT, done=True)
    print("\n An error was encountered while exporting data!")
    raise e
    
stopLoading(lE, lT)

#%% Cleaning up...
del SS_end, SS_start, lE, lT

#%% ARCHIVE...
"""
fig5,ax5 = plt.subplots(1, 1, dpi=600)
ax5.plot(np.linspace(0,3750,100),np.linspace(0,3750,100),color = "black",linestyle="dashed")
ax5.plot(reg_model_diff_low["Actual value"],reg_model_diff_low["Predicted value"],color="green")
ax5.fill_between(reg_model_diff_low["Actual value"],reg_model_diff_low["Predicted value"]-RMSlow,reg_model_diff_low["Predicted value"]+RMSlow,
                 color = "green",alpha = 0.25)
ax5.plot(reg_model_diff_med["Actual value"],reg_model_diff_med["Predicted value"],color = "blue")
ax5.fill_between(reg_model_diff_med["Actual value"],reg_model_diff_med["Predicted value"]-RMSmid,reg_model_diff_med["Predicted value"]+RMSmid,
                 color = "blue",alpha = 0.25)
ax5.plot(reg_model_diff_high["Actual value"],reg_model_diff_high["Predicted value"],color = "orange")
ax5.fill_between(reg_model_diff_high["Actual value"],reg_model_diff_high["Predicted value"]-RMShigh,reg_model_diff_high["Predicted value"]+RMShigh,
                 color = "orange",alpha = 0.25)
plt.show()

---- Plotting predicted and delivered methane concentrations across calibration setpoints ----
# Defining setpoint error
setpointerror = 0.0199
fig3,ax3 = plt.subplots(1, 1, figsize=(9, 5), dpi = 600)              # Generating plot
ax3.plot(reg_model_diff_low.index, reg_model_diff_low["Predicted value"], label = "Low Range", color = "mediumseagreen")         # Plotting low range 
ax3.fill_between(reg_model_diff_low.index, reg_model_diff_low["Predicted value"] - RMSlow,
                  reg_model_diff_low["Predicted value"] + RMSlow,color = 'mediumseagreen', alpha = 0.25, zorder=-1)     # Filling in 
ax3.plot(reg_model_diff_med.index,reg_model_diff_med["Predicted value"], label = "Mid Range", color = "orchid")         # Plotting mid range
ax3.fill_between(reg_model_diff_med.index, reg_model_diff_med["Predicted value"] - RMSmid,
                  reg_model_diff_med["Predicted value"] + RMSmid,color = 'orchid', alpha = 0.25, zorder=-1)
ax3.plot(reg_model_diff_high.index, reg_model_diff_high["Predicted value"], label = "High Range", color = "orange")     # Plotting high range
ax3.fill_between(reg_model_diff_high.index, reg_model_diff_high["Predicted value"] - RMShigh,
                  reg_model_diff_high["Predicted value"] + RMShigh,color = 'orange', alpha = 0.25, zorder=-1)
# Plotting setpoint line
ax3.plot(setpointavg.index, setpointavg["Setpoint"], color = "black", label = "Setpoints", linestyle = ':', alpha = 0.75)
ax3.fill_between(setpointavg.index, setpointavg["Setpoint"] + setpointerror, setpointavg["Setpoint"] - setpointerror,
                  alpha=0.25, color = "b", zorder=-1)
# Creating legend and setting labels
ax3.legend()
ax3.set_xlabel("Setpoint Index", fontsize=12)
ax3.set_ylabel("Setpoint Concentration of Methane $[ppm]$", fontsize=12)
ax3.grid(True, axis='y', zorder=-1, color="grey", alpha=0.5)
fig3.suptitle("Low, Medium, and High Regression Response to Setpoints", fontsize=15, fontweight='bold')
ax3.set_xlim(left=np.min(reg_model_diff_low.index), right=np.max(reg_model_diff_high.index))
plt.show()

#%% Boxplot to see sensor ranges for calibration vs field deployment...
# What variable do you want to plot?
plotVariable = "BO TGS2602"

# Setting times for calibration and field deployment
TS = MOX.index
# Create a copy for the calibration data and add a 'Period' column
cal = MOX.loc[(TS >= startTime) & (TS <= pd.to_datetime("2025-06-03 23:23:00"))].copy()
cal['Period'] = 'Calibration'

# Create a copy for the field data and add a 'Period' column
field = MOX.loc[(TS >= pd.to_datetime("2025-06-16 15:17:00")) & (TS <= endTime)].copy()
field['Period'] = 'Field'

fig, axs = plt.subplots(1, 2, figsize=(10, 6), dpi=600, sharey=False)

# --- Plot 1: Calibration Data ---
sns.boxplot(
    data=cal,
    y=plotVariable,
    ax=axs[0],           # Plot on the first axis
    color=sns.color_palette('Set2')[0] # Manually select a color
)
axs[0].set_title('Calibration Period', fontsize=14)
axs[0].set_ylabel('Sensor Readings (mV)', fontsize=12)
axs[0].set_xlabel('') # We can remove the x-label if the title is clear
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
axs[1].set_ylabel('') # Remove redundant y-label
axs[1].set_xlabel('')
axs[1].grid(True, linestyle='--', which='major', color='grey', alpha=0.5)

# Add a main title for the entire figure
fig.suptitle(f"{plotVariable} Reading Distribution", fontsize=16, y=0.98)

# Adjust layout to prevent titles from overlapping
plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust rect to make space for suptitle
plt.show()
"""