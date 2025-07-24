"""
Model Testing :)
Jessica Goff
06.24.25
----------------------------------------------------------------- DESCRIPTION -----------------------------------------------------------------
This code is used to test regression model accuracy for each calibration range. It iterates over a train/test split a set number of times and averages
the RMSE and R² values for evaluation.

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
        pcbNum: Boron sensor you are using
        iterNum: Number of times to iterate over model testing
    
OUTPUT:
    Averaged RMSE and R² values for each range and model.
"""

# Importing loading UI essentials...
from Python.loadingUI import startLoading, stopLoading
lE, lT = startLoading(message = "Importing libraries")

# Importing essential libraries...
from Python.sheets_puller import sheetPuller
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
stopLoading(lE, lT)



#%% ------------------------------------------------------------------ User-Defined Variables ------------------------------------------------------------------

defaultPath = "MetaData_060325.csv"                     # Setpoints file from calibration

pcbNum = 3              # What Boron sensor are you using (3, 4, or 5)?

# Setting start & end times
# Selected times isolate the calibration period
startTime = pd.to_datetime("2025-06-03 15:23:00")      # Must be within selected dataset
endTime = pd.to_datetime("2025-06-09 16:17:00")        # Must be in [yyyy-mm-dd hh-mm-ss] format

# Setting steady state time range within start & end times
SS_start = 11       # [min]
SS_end   = 15       # [min]

# Defining the number of iterations for the model testing
iterNum = 1000


#%% Importing data...
lE, lT = startLoading(message = "Importing data", t=0.25)

# Importing data from Google sheets
try: MOX = sheetPuller(f"Boron {pcbNum}", "06.03.25 Calibration")
except Exception as e:
    stopLoading(lE, lT, done=False)
    print("\nAn error was encountered while importing data!")
    raise e

# Importing setpoint file
Setpoints = pd.read_csv(defaultPath)                        # defaultPath specified at top of code

stopLoading(lE, lT)


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
    "Main TGS2600 (mV)", "Main TGS2602 (mV)", "Main TGS2611 (mV)",
    "Main EC_Worker (mV)", "Main EC_Aux (mV)", "Temperature (C)",
    "Pressure (hPa)", "Humidity (%)", "GasResistance (Ohms)", "FlowRate (slm)", "FlowTemp (C)"], errors = 'ignore', inplace = True)

# Renaming columns to remove units
MOX.rename(columns = {"BO TGS2600 (mV)" : "BO TGS2600", "BO TGS2602 (mV)" : "BO TGS2602", "BO TGS2611 (mV)" : "BO TGS2611", "BO EC_Worker (mV)" : "BO EC_Worker",
                      "BO EC_Aux (mV)" : "BO EC_Aux", "SGX_Analog (mV)" : "SGX_Analog", "SGX_Digital (ppm)" : "SGX_Digital", "BO Temperature (C)" : "BO Temperature",
                      "BO Pressure (hPa)" : "BO Pressure", "BO Humidity (%)" : "BO Humidity", "BO GasResistance (Ohms)" : "BO GasResistance"}, inplace = True)

# Cleaning data values and resampling for plotting
MOX = (MOX.mask(MOX == 404, np.nan).infer_objects(copy = False)
       .dropna().resample("60s").last().dropna())


#%% Cleaning Setpoints data...

#Setpoints.columns = Setpoints.columns.str.strip()                                       # Removing extraneous spaces
Setpoints = Setpoints.rename(columns = {'Date [mm/dd] Start Time [hh:mm:ss]': 'time'})  # Renaming time column

# Converting 'time' column to datetime and coercing errors
Setpoints['time'] = pd.to_datetime(Setpoints['time'], format="mixed")  
Setpoints = Setpoints.set_index('time')                         # Dropping nan rows and setting time as index

# Isolating start and end times in Setpoints
Setpoints = Setpoints[(Setpoints.index > startTime) & (Setpoints.index < endTime)]


#%% Splitting Setpoints into Training and Testing sets

print("Splitting setpoint events into training and testing sets...")

# Splitting the Setpoints DataFrame, which represents the individual experimental runs
# test_size=0.2 means 20% of the events will be reserved for testing.
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """
    Fits a model and evaluates its performance.
    
    Args:
        model: The regression model instance to train.
        X_train: Training feature data.
        y_train: Training target data.
        X_test: Testing feature data.
        y_test: Testing target data.
        
    Returns:
        A tuple containing (rmse, r2, predictions).
    """
    # Fit the model to the training data
    model.fit(X_train, y_train)
    
    # Make predictions on the test data
    predictions = model.predict(X_test)
    
    # Calculate performance metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    return rmse, r2, predictions


def create_setpoint_averages(setpoints_df, mox_df, ss_start, ss_end):
    """
    Processes a setpoints dataframe to create averaged sensor readings.

    Args:
        setpoints_df: A DataFrame of setpoint events (either training or testing).
        mox_df: The main MOX sensor data DataFrame.
        ss_start (int): The start minute for the steady-state window.
        ss_end (int): The end minute for the steady-state window.

    Returns:
        A new DataFrame with averaged sensor data for each setpoint event.
    """
    results = []
    for timestamp, row in setpoints_df.iterrows():
        try:
            window_start = timestamp + timedelta(minutes=ss_start)
            window_end = timestamp + timedelta(minutes=ss_end)
            window_data = mox_df.loc[window_start:window_end]

            if not window_data.empty:
                sensor_means = window_data.mean()
                new_row_dict = sensor_means.to_dict()
                new_row_dict['H2S'] = row['C_H2S [ppm]']
                new_row_dict['Setpoint'] = row['C_CH4 [ppm]']
                results.append(new_row_dict)
            else:
                pass
                #print(f"No data found for window around {timestamp.strftime('%Y-%m-%d %H:%M:%S')}. Skipping.")

        except Exception as e:
            pass
            #print(f"An unexpected error occurred for timestamp {timestamp}: {e}")

    # Create the final dataframe from our list of processed dictionaries
    final_df = pd.DataFrame(results)
    
    # Sort and reset the index based on setpoints
    final_df = (
        final_df.sort_values(by="Setpoint", ascending=True)
                .reset_index(drop=True))
    
    return final_df

# Creating interaction features and ratios
def ratioFeatures(df):
    """
    Takes a dataframe and adds new interaction and ratio features.
    """
    # Create a copy to avoid changing the original DataFrame
    df_out = df.copy()
    
    # Calculate and add the new feature columns
    df_out['R11_00'] = df_out['BO TGS2611'] / (df_out['BO TGS2600'])
    df_out['Temp_Humid'] = df_out['BO Temperature'] * df_out['BO Humidity']
    df_out['Temp_2611'] = df_out['BO TGS2611'] * df_out['BO Temperature']
    df_out['SGX_Temp'] = df_out['SGX_Digital'] * df_out['BO Temperature']
    
    return df_out

featuresLow = ["BO TGS2600", "BO TGS2611", "R11_00", "Temp_Humid"]
featuresMed = ["BO TGS2611", "BO Humidity", "BO Temperature"]
featuresHigh = ["SGX_Digital", "BO Temperature", "BO TGS2611", "BO Humidity", "SGX_Temp", "Temp_2611"]


#%% Loop stuff
lE, lT = startLoading(message="Performing calibration..")

# Creating list to store results
all_results = []

# for loop to iterate over regression models
for i in range(iterNum):
    train_setpoints, test_setpoints = train_test_split(
        Setpoints,
        test_size=0.2,
    )

    # --- Create the training and testing setpointavg DataFrames using our new function ---
    setpointavg_train = create_setpoint_averages(train_setpoints, MOX, SS_start, SS_end)
    setpointavg_test = create_setpoint_averages(test_setpoints, MOX, SS_start, SS_end)
    setpointavg_train = ratioFeatures(setpointavg_train)
    setpointavg_test = ratioFeatures(setpointavg_test)


    #% --- Data Wrangling ---
    lowTrain = setpointavg_train[setpointavg_train["Setpoint"] <= 71]
    medTrain = setpointavg_train[(setpointavg_train["Setpoint"] >= 60) & (setpointavg_train["Setpoint"] <= 1000)]
    highTrain = setpointavg_train[setpointavg_train["Setpoint"] >= 600]
    lowTest = setpointavg_test[setpointavg_test["Setpoint"] <= 71]
    medTest = setpointavg_test[(setpointavg_test["Setpoint"] > 71) & (setpointavg_test["Setpoint"] <= 750)]
    highTest = setpointavg_test[setpointavg_test["Setpoint"] > 750]
    
    # Format: (Range Name, Training Data, Testing Data, Feature List)
    data_ranges = [
        ("Low", lowTrain, lowTest, featuresLow),
        ("Medium", medTrain, medTest, featuresMed),
        ("High", highTrain, highTest, featuresHigh)
    ]
    
    
    # Polynomial models
    polynomial_model_deg2 = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('linear', linear_model.LinearRegression())
    ])

    # You can also experiment with a higher degree
    polynomial_model_deg3 = Pipeline([
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('linear', linear_model.LinearRegression())
    ])
    
    # Creating a dictionary of models to test
    # The parameters (e.g., alpha, n_estimators) can be tuned for better performance.
    models_to_test = {
        "Linear Regression": linear_model.LinearRegression(),
        "Polynomial Regression (Deg 2)": polynomial_model_deg2,
        "Polynomial Regression (Deg 3)": polynomial_model_deg3,
        "Gradient Boosting": HistGradientBoostingRegressor(max_leaf_nodes=10)
    }
    
    
    #% --- Run Experiment ---
    # Loop over each data range
    for name, train_df, test_df, features in data_ranges:
        #print(f"--- Evaluating models for {name} Range ---")
        
        # Prepare the data for the current range
        X_train = train_df[features]
        y_train = train_df["Setpoint"]
        X_test = test_df[features]
        y_test = test_df["Setpoint"]

        # Loop over each model
        for model_name, model in models_to_test.items():
            # Train and evaluate the model using our function
            rmse, r2, _ = train_and_evaluate(model, X_train, y_train, X_test, y_test)
            
            # Store the results
            all_results.append({
                "Range": name,
                "Model": model_name,
                "RMSE": rmse,
                "R^2 Score": r2
            })
    if (i+1) % 10 == 0: print(f"Progress: ({i+1}/{iterNum})")

    
# Convert the list of results into a DataFrame for easy viewing
resultsDf = pd.DataFrame(all_results)
# Averaging the results in RMSE and R² values
average_results_df = resultsDf.groupby(["Range", "Model"])[["RMSE", "R^2 Score"]].mean()

stopLoading(lE, lT)

#%% Printing final results...
print("\n\n------ MODEL COMPARISON REPORT ------")
print(average_results_df.to_string())


#%% R² Boxplot...
# Defining order for range boxplots
range_order = ["Low", "Medium", "High"]

# Creating the figure and axes for the plot
fig1, ax1 = plt.subplots(figsize=(8, 7), dpi=600)

# Drawing the grouped boxplot
sns.boxplot(
    data=resultsDf,
    x="Range",
    y="R^2 Score",
    hue="Model",
    ax=ax1,
    order=range_order,  # Apply the specified order here
    palette="Set2",     # A nice color scheme for visual distinction
    showfliers = False      
)

# Customizing the plot for clarity and presentation
ax1.set_title("Model R² Score Distribution by Calibration Range", fontsize=16, pad=20)
ax1.set_ylabel("R² Score", fontsize=12)
ax1.set_xlabel("Calibration Range", fontsize=12)

# Adding a horizontal grid for easier reading of y-axis values
ax1.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
ax1.set_axisbelow(True)  # Puts gridlines behind the plot elements

# Adjusting & placing legend
ax1.legend(title="Model", loc='lower right')

# Ensuring all plot elements fit without overlapping
plt.tight_layout(rect=[0, 0, 0.9, 1])

# Displaying the final plot
plt.show()


#%% Archived plots...
"""
#%% CH4 Residuals Plot...

# --- Step 1: Initialize variables to track the best model run ---
best_overall_rmse = float('inf')
best_run_residuals_data = None
n_iterations = 100

print(f"Running {n_iterations} iterations to find the optimal train/test split...")
lE, lT = startLoading(message=f"Optimizing models... Please wait for {n_iterations} iterations to complete.")

# --- Step 2: Loop 100 times to find the best train/test split ---
for i in range(n_iterations):
    # The lines that caused the error have been removed from the loop.
    
    # Perform a new random split of the setpoint data
    train_setpoints, test_setpoints = train_test_split(Setpoints, test_size=0.2)
    
    # Create the averaged dataframes based on the new split
    setpointavg_train = create_setpoint_averages(train_setpoints, MOX, SS_start, SS_end)
    setpointavg_test = create_setpoint_averages(test_setpoints, MOX, SS_start, SS_end)

    # Wrangle the new data into Low, Medium, and High ranges
    lowTrain = setpointavg_train[setpointavg_train["Setpoint"] <= 30]
    medTrain = setpointavg_train[(setpointavg_train["Setpoint"] > 30) & (setpointavg_train["Setpoint"] <= 200)]
    highTrain = setpointavg_train[setpointavg_train["Setpoint"] > 200]
    lowTest = setpointavg_test[setpointavg_test["Setpoint"] <= 30]
    medTest = setpointavg_test[(setpointavg_test["Setpoint"] > 30) & (setpointavg_test["Setpoint"] <= 200)]
    highTest = setpointavg_test[setpointavg_test["Setpoint"] > 200]

    # Define the models for this run
    model_map = {
        "Low":    {"model": polynomial_model_deg2, "train_df": lowTrain,  "test_df": lowTest,  "features": featuresLow},
        "Medium": {"model": polynomial_model_deg2, "train_df": medTrain,  "test_df": medTest,  "features": featuresMed},
        "High":   {"model": polynomial_model_deg3, "train_df": highTrain, "test_df": highTest, "features": featuresHigh}
    }
    
    current_run_rmses = []
    current_run_residuals = []
    
    # Train models and calculate performance for the CURRENT run
    for name, details in model_map.items():
        X_train, y_train = details["train_df"][details["features"]], details["train_df"]["Setpoint"]
        X_test, y_test = details["test_df"][details["features"]], details["test_df"]["Setpoint"]

        if not X_test.empty and not X_train.empty:
            model = details["model"]
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            current_run_rmses.append(rmse)
            
            residuals = y_test - predictions
            for true_val, res in zip(y_test, residuals):
                current_run_residuals.append({
                    "True Value (ppm)": true_val, "Residual Error (ppm)": res, "Range": name
                })
    
    # Check if this run was better than the best one seen so far
    if current_run_rmses:
        current_avg_rmse = np.mean(current_run_rmses)
        if current_avg_rmse < best_overall_rmse:
            best_overall_rmse = current_avg_rmse
            best_run_residuals_data = current_run_residuals.copy()

stopLoading(lE, lT)
print(f"\nOptimization complete. Best Average RMSE found: {best_overall_rmse:.4f}")

# --- Step 3: Convert the best run's data into a DataFrame ---
residuals_df = pd.DataFrame(best_run_residuals_data)

# --- Step 4: Define the custom x-axis scale using the BEST run's data ---
if not residuals_df.empty:
    setpoint_anchors = np.unique(residuals_df["True Value (ppm)"].values)
    min_val, max_val = setpoint_anchors.min(), setpoint_anchors.max()
    padding_amount = (max_val - min_val) * 0.05
    scale_anchors = np.unique(np.concatenate(([min_val - padding_amount], setpoint_anchors, [max_val + padding_amount])))
    linear_positions = np.linspace(0, 1, len(scale_anchors))

    def forward_transform(values): return np.interp(values, scale_anchors, linear_positions)
    def inverse_transform(ranks): return np.interp(ranks, linear_positions, scale_anchors)

    # --- Step 5: Generate the final plot for the BEST performing run ---
    fig_res, ax_res = plt.subplots(figsize=(11, 7), dpi=150)
    ax_res.set_xscale('function', functions=(forward_transform, inverse_transform))

    sns.scatterplot(data=residuals_df, x="True Value (ppm)", y="Residual Error (ppm)",
                    hue="Range", style="Range", s=70, ax=ax_res, palette="viridis")

    ax_res.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error Line')
    ax_res.set_title("Optimized Residuals Plot (Best of 100 Runs)", fontsize=18, pad=20)
    ax_res.set_xlabel("True CH4 Concentration (Setpoint) [ppm]", fontsize=12)
    ax_res.set_ylabel("Residual Error (True - Predicted) [ppm]", fontsize=12)
    ax_res.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_res.set_xticks(setpoint_anchors)

    handles, labels = ax_res.get_legend_handles_labels()
    zero_line_idx = labels.index('Zero Error Line')
    handles.append(handles.pop(zero_line_idx))
    labels.append(labels.pop(zero_line_idx))
    ax_res.legend(handles=handles, labels=labels, title='Calibration Range')

    plt.tight_layout()
    plt.show()
else:
    print("Could not generate residuals plot. No successful runs were completed.")


#%% Histogram of RMSE values...
# ==================================================================
#         RMSE Histogram Subplots by Calibration Range
# ==================================================================
print("\nGenerating RMSE histogram subplots...")

# --- Step 1: Set up the figure and axes for 3 subplots ---
# We create one row with three columns. `sharey=True` makes comparing heights easier.
fig, axes = plt.subplots(1, 3, figsize=(20, 7), dpi=600)

# Get the list of models you tested
model_names = resultsDf['Model'].unique()
range_names = ["Low", "Medium", "High"]

# --- Step 2: Loop through each range and its corresponding subplot axis ---
for ax, range_name in zip(axes, range_names):
    # Filter the main results DataFrame for the current range
    range_data = resultsDf[resultsDf['Range'] == range_name]
    
    # --- Step 3: Plot a histogram for each model on the current subplot ---
    for model_name in model_names:
        # Filter the range-specific data for the current model
        model_data = range_data[range_data['Model'] == model_name]
        
        # Plot the histogram using Seaborn
        sns.histplot(
            data=model_data,
            x='RMSE',
            kde=True,           # Adds a smooth density line over the histogram
            ax=ax,
            label=model_name,
            element="step",     # 'step' style is great for overlapping histograms
            common_norm=False   # Normalize each histogram independently
        )
    
    # --- Step 4: Customize each subplot ---
    ax.set_title(f'RMSE Distribution for {range_name} Range', fontsize=14, pad=10)
    ax.set_xlabel('Root Mean Squared Error (RMSE)', fontsize=11)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

# Add a main, overarching title to the entire figure
fig.suptitle('Comparison of Model RMSE Distributions by Range', fontsize=20, fontweight='bold')

# Adjust the layout to prevent titles from overlapping
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()
fig.savefig(f"Figures/Boron{pcbNum}HistogramRMSE.png", dpi = 600)
"""