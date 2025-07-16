"""
Methane Analysis
03.06.2025
Jessica Goff
"""

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from matplotlib.backends.backend_pdf import PdfPages
from sheets_puller import sheetPuller


#%% -------------------------------------------- USER INPUTS --------------------------------------------
pcbNum = 8                                  # Which Boron do you want to use?

sheet = "Data"                              # Sheet name inside google spreadsheet

# Setting start & end times
startTime = pd.to_datetime("2025-07-02 10:35:00")      # Must be within selected dataset
endTime = pd.to_datetime("2025-07-08 18:00:00")        # Must be in [yyyy-mm-dd hh-mm-ss] format


#%% Importing and cleaning data...

# Importing sheet
data = sheetPuller(sheet, f"Boron_{pcbNum}")

print("Cleaning and preparing data...")

# Converting 'Time' column and coercing errors to NaT (Not a Time)
data['Time'] = pd.to_datetime(data['Time'], format='%Y-%m-%d %H:%M:%S', errors = 'coerce')

# Dropping rows with invalid dates and setting time as the index
data = data.set_index('Time')
data = data.sort_index()

# Removing rows with the boot-up timestamp
data = data[data.index != pd.Timestamp("1999-12-31 17:00:12")]

# Filtering to the date range of interest
print(f"\n---- Data Timeframe Adjustments ----\nData size before filtering: {len(data)}")
# Checking if the date range does not exist
if not data.empty:
    try:
        data = data.loc[(data.index >= startTime) & (data.index <= endTime)]
    except KeyError:
        print("Warning: The specified startTime or endTime was not found in the data. Skipping date range filter.")
print(f"Data size after filtering: {len(data)}\n")

# Dropping irrelevant columns
data.drop(columns = ["FlowRate (slm)", "FlowTemp (C)"], errors = 'ignore', inplace = True)

# Renaming columns to remove units
data.rename(columns = {"Main TGS2600 (mV)": "TGS2600", "Main TGS2602 (mV)": "TGS2602", "Main TGS2611 (mV)": "TGS2611", 
                       "Main EC_Worker (mV)" : "EC_Worker", "Main EC_Aux (mV)" : "EC_Aux", "Temperature (C)" : "Temperature", 
                       "Pressure (hPa)" : "Pressure", "Humidity (%)" : "Humidity", "GasResistance (Ohms)" : "aGas",
                       "BO TGS2600 (mV)" : "BO TGS2600", "BO TGS2602 (mV)" : "BO TGS2602", "BO TGS2611 (mV)" : "BO TGS2611", 
                       "BO EC_Worker (mV)" : "BO EC_Worker", "BO EC_Aux (mV)" : "BO EC_Aux", "SGX_Analog (mV)" : "SGX_Analog", 
                       "SGX_Digital (ppm)" : "SGX_Digital", "BO Temperature (C)" : "BO Temperature", "BO Pressure (hPa)" : "BO Pressure", 
                       "BO Humidity (%)" : "BO Humidity", "BO GasResistance (Ohms)" : "BO GasResistance"}, inplace = True)

# List of columns that must be numeric values
numericCols = [
    "TGS2600", "TGS2602", "TGS2611", "EC_Worker", "EC_Aux", "Temperature", 
    "Pressure", "Humidity", "aGas", "BO TGS2600", "BO TGS2602", "BO TGS2611", 
    "BO EC_Worker", "BO EC_Aux", "SGX_Analog", "SGX_Digital", "BO Temperature", 
    "BO Pressure", "BO Humidity", "BO GasResistance"
]

# Looping through the list of columns to convert each column to a numeric type
for col in numericCols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Dropping rows with Nan values
data.dropna(inplace=True)


#%% Smoothing data to plot...

print("Smoothing data...")

# Creating a new DataFrame for the smoothed data to keep the original safe and unedited
smoothed_data = data.copy()
smoothed_data["TGS2600"] = gaussian_filter1d(data["TGS2600"], sigma=15)
smoothed_data["BO TGS2600"] = gaussian_filter1d(data["BO TGS2600"], sigma=15)
smoothed_data["TGS2602"] = gaussian_filter1d(data["TGS2602"], sigma=15)
smoothed_data["BO TGS2602"] = gaussian_filter1d(data["BO TGS2602"], sigma=15)
smoothed_data["TGS2611"] = gaussian_filter1d(data["TGS2611"], sigma=20)
smoothed_data["BO TGS2611"] = gaussian_filter1d(data["BO TGS2611"], sigma=20)
smoothed_data["Temperature"] = gaussian_filter1d(data["Temperature"], sigma=25)
smoothed_data["BO Temperature"] = gaussian_filter1d(data["BO Temperature"], sigma=25)
smoothed_data["Humidity"] = gaussian_filter1d(data["Humidity"], sigma=10)
smoothed_data["BO Humidity"] = gaussian_filter1d(data["BO Humidity"], sigma=10)


#%% Normalizing data against time
t0 = data.index.min()                                                     # Earliest date in dTime dataset
nTime = (data.index - t0).total_seconds()                                 # Time normalized by seconds from earliest date to latest
del t0                                                                    # Deleting intermediate variable


#%% Plotting any value over a set range of dates
def plotRange(tStart, tRange):
    """
    
    Plots data over a range of days given a starting date and number of days to plot.
    ----------
    tStart : Starting date in yyyy-mm-dd format
    tRange : Number of days to plot
    -------
    """
    tStart = pd.Timestamp(tStart).date()
    if tStart not in data.index.date:
        raise ValueError("The date you are looking for does not exist in the data :(")
    t0 = data.index[data.index.date == tStart][0]
    startIndex = np.where(data.index == t0)[0][0]
    endIndex = np.where(data.index.date > (t0.date() + pd.Timedelta(days = tRange-1)))[0]
    if endIndex.size == 0:
        tRange = range(startIndex, data.index.size)
        time = data.index[tRange]
    else:
        endIndex = endIndex[0]
        tRange = range(startIndex, endIndex)
        time = data.index[tRange]
    nTime = (time - t0).total_seconds()
    if (time.date[-1] - time.date[0]) < pd.Timedelta(days = 5):
        nTime = nTime/3600
        xLabel = "Time $[Hours]$"
    else:
        nTime = nTime/(3600*24)
        xLabel = "Time $[Days]$"
    
    counter = 0
    figs = {}
    
    fig = f"fig{counter}"
    ax = f"ax{counter}"
    
    fig, ax = plt.subplots(1, 1, figsize = (8.5, 4.3), dpi = 1200)
    ax.plot(nTime, smoothed_data["TGS2600"].iloc[tRange], linewidth = 0.7)
    ax.plot(nTime, smoothed_data["BO TGS2600"].iloc[tRange], linewidth = 0.7)
    ax.set_xlim(left = 0)
    ax.set_xlabel(xLabel, fontsize = 13)
    ax.set_ylabel("Reading $[mV]$", fontsize = 13)
    ax.legend(["Ambient", "Internal"], title = "Sensors")
    ax.set_title(f"Boron {pcbNum} TGS2600", fontsize = 16, fontweight = 'bold')
    
    figs[counter] = fig
    counter += 1

    fig, ax = plt.subplots(1, 1, figsize = (8.5, 4.3), dpi = 1200)
    ax.plot(nTime, smoothed_data["TGS2602"].iloc[tRange], linewidth = 0.7)
    ax.plot(nTime, smoothed_data["BO TGS2602"].iloc[tRange], linewidth = 0.7)
    ax.set_xlim(left = 0)
    # Scaling graph
    #ax.set_ylim(top = 2000)
    ax.set_xlabel(xLabel, fontsize = 13)
    ax.set_ylabel("Reading $[mV]$", fontsize = 13)
    ax.legend(["Ambient", "Internal"], title = "Sensors")
    ax.set_title(f"Boron {pcbNum} TGS2602", fontsize = 16, fontweight = 'bold')
    
    figs[counter] = fig
    counter += 1

    fig, ax = plt.subplots(1, 1, figsize = (8.5, 4.3), dpi = 1200)
    ax.plot(nTime, smoothed_data["TGS2611"].iloc[tRange], linewidth = 0.6)
    ax.plot(nTime, smoothed_data["BO TGS2611"].iloc[tRange], linewidth = 0.6)
    ax.set_xlim(left = 0)
    ax.set_xlabel(xLabel, fontsize = 13)
    ax.set_ylabel("Reading $[mV]$", fontsize = 13)
    ax.legend(["Ambient", "Internal"], title = "Sensors")
    ax.set_title(f"Boron {pcbNum} TGS2611", fontsize = 16, fontweight = 'bold')
    
    figs[counter] = fig
    counter += 1

    fig, ax = plt.subplots(1, 1, figsize = (8.5, 4.3), dpi = 1200)
    ax.plot(nTime, smoothed_data["Temperature"].iloc[tRange], linewidth = 0.6)
    ax.plot(nTime, smoothed_data["BO Temperature"].iloc[tRange], linewidth = 0.6)
    ax.set_xlim(left = 0)
    ax.set_xlabel(xLabel, fontsize = 13)
    ax.set_ylabel("Temperature $[°C]$", fontsize = 13)
    ax.legend(["Ambient", "Internal"], title = "Sensors")
    ax.set_title(f"Boron {pcbNum} Temperature $[°C]$", fontsize = 16, fontweight = 'bold')
    
    figs[counter] = fig
    counter += 1

    fig, ax = plt.subplots(1, 1, figsize = (8.5, 4.3), dpi = 1200)
    ax.plot(nTime, smoothed_data["Humidity"].iloc[tRange], linewidth = 0.6)
    ax.plot(nTime, smoothed_data["BO Humidity"].iloc[tRange], linewidth = 0.6)
    ax.set_xlim(left = 0)
    ax.set_xlabel(xLabel, fontsize = 13)
    ax.set_ylabel(f"Boron {pcbNum} Relative Humidity [%]", fontsize = 13)
    ax.legend(["Ambient", "Internal"], title = "Sensors")
    ax.set_title(f"Boron {pcbNum} Humidity [%]", fontsize = 16, fontweight = 'bold')
    
    figs[counter] = fig
    counter += 1
    
    fig, ax = plt.subplots(1, 1, figsize = (10.5, 6.25), dpi = 1200)
    ax.plot(nTime, smoothed_data["BO TGS2600"].iloc[tRange], linewidth = 0.65)
    ax.plot(nTime, smoothed_data["BO TGS2611"].iloc[tRange], linewidth = 0.65)
    ax.set_xlim(left = 0)
    ax.set_xlabel(xLabel, fontsize = 14)
    ax.set_ylabel("Reading $[mV]$", fontsize = 14)
    ax.legend(["Internal 2600", "Internal 2611", "Ambient 2611", "Box 2611"], title = "Sensors")
    ax.set_title(f"Boron {pcbNum} TGS2600 vs. TGS2611", fontsize = 17.5, fontweight = 'bold')
    
    figs[counter] = fig
    counter += 1
    
    fig, ax = plt.subplots(1, 1, figsize = (8.5, 4.3), dpi = 1200)
    ax.plot(nTime, smoothed_data["SGX_Digital"].iloc[tRange], linewidth = 0.6)
    ax.set_xlim(left = 0)
    ax.set_xlabel(xLabel, fontsize = 13)
    ax.set_ylabel("SGX Digital $[ppm]$", fontsize = 13)
    ax.set_title(f"Boron {pcbNum} SGX Digital Readings", fontsize = 16, fontweight = 'bold')
    
    figs[counter] = fig

    # Return plots for exporting
    return figs


#%% Plotting different date ranges for the same variable
def compareRange(t1Start, t2Start, tRange):
    """
    Plots data over two ranges of days given each starting date and number of days to plot.
    ----------
    t1Start : Starting date #1 in yyyy-mm-dd format
    t2Start: Starting date #2 in yyyy-mm-dd format
    tRange : Number of days to plot
    -------
    """
    t1Start = pd.Timestamp(t1Start).date()
    t2Start = pd.Timestamp(t2Start).date()
    if t1Start not in data.index.date:
        raise ValueError(f"The date you entered (t1Start = {t1Start}) does not exist in the data :(")
    if t2Start not in data.index.date:
        raise ValueError(f"The date you entered (t2Start = {t2Start}) does not exist in the data :(")
    t01 = data.index[data.index.date == t1Start][0]
    t02 = data.index[data.index.date == t2Start][0]
    startIndex1 = np.where(data.index == t01)[0][0]
    endIndex1 = np.where(data.index.date > (t01.date() + pd.Timedelta(days = tRange-1)))[0]
    if endIndex1.size == 0:
        tRange1 = range(startIndex1, data.index.size)
        time1 = data.index[tRange1]
    else:
        endIndex1 = endIndex1[0]
        tRange1 = range(startIndex1, endIndex1)
        time1 = data.index[tRange1]    
    startIndex2 = np.where(data.index == t02)[0][0]
    endIndex2 = np.where(data.index.date > (t02.date() + pd.Timedelta(days = tRange-1)))[0]
    if endIndex2.size == 0:
        tRange2 = range(startIndex2, data.index.size)
        time2 = data.index[tRange2]
    else:
        endIndex2 = endIndex2[0]
        tRange2 = range(startIndex2, endIndex2)
        time2 = data.index[tRange2]    
    nTime1 = (time1 - t01).total_seconds()
    nTime2 = (time2 - t02).total_seconds()
    if (tRange < 5):
        nTime1 = nTime1/3600
        nTime2 = nTime2/3600
        xLabel = "Time $[Hours]$"
        tRange = tRange*24
    else:
        nTime1 = nTime1/(3600*24)
        nTime2 = nTime2/(3600*24)
        xLabel = "Time $[Days]$"

    figs = {}
    counter = 0
    fig = f"fig{counter}"
    axa = f"ax{counter}a"
    axb = f"ax{counter}b"
    
    fig, (axa, axb) = plt.subplots(2, 1, figsize = (10.5, 6.25), dpi = 1200)
    axa.plot(nTime1, smoothed_data["TGS2600"].iloc[tRange1], linewidth = 0.55)
    axa.plot(nTime1, smoothed_data["BO TGS2600"].iloc[tRange1], linewidth = 0.55)
    axa.set_ylabel("Reading $[µV]$", fontsize = 13)
    axa.set_xlim(left=0, right = tRange)
    #axa.set_ylim(top = 1500)
    axb.plot(nTime2, smoothed_data["TGS2600"].iloc[tRange2], linewidth = 0.55)
    axb.plot(nTime2, smoothed_data["BO TGS2600"].iloc[tRange2], linewidth = 0.55)
    axb.legend(["Ambient", "Box"], title = "Sensors")
    axb.set_xlabel(xLabel, fontsize = 13)
    axb.set_ylabel("Reading $[µV]$", fontsize = 13)
    axb.set_xlim(left=0, right = tRange)
    #axb.set_ylim(top = 1500)
    axa.set_title(f"Boron {pcbNum} TGS2600", fontsize = 16, fontweight = 'bold')
    
    figs[counter] = fig
    counter += 1    

    fig, (axa, axb) = plt.subplots(2, 1, figsize = (10.5, 6.25), dpi = 1200)
    axa.plot(nTime1, smoothed_data["TGS2611"].iloc[tRange1], linewidth = 0.55)
    axa.plot(nTime1, smoothed_data["BO TGS2611"].iloc[tRange1], linewidth = 0.55)
    axa.set_ylabel("Reading $[µV]$", fontsize = 13)
    axa.set_xlim(left=0, right = tRange)
    #axa.set_ylim(top = 850)
    axb.plot(nTime2, smoothed_data["TGS2611"].iloc[tRange2], linewidth = 0.55)
    axb.plot(nTime2, smoothed_data["BO TGS2611"].iloc[tRange2], linewidth = 0.55)
    axb.legend(["Ambient", "Box"], title = "Sensors")
    axb.set_xlabel(xLabel, fontsize = 13)
    axb.set_ylabel("Reading $[µV]$", fontsize = 13)
    axb.set_xlim(left=0, right = tRange)
    #axb.set_ylim(top = 850)
    axa.set_title(f"Boron {pcbNum} TGS2611", fontsize = 16, fontweight = 'bold')
    
    figs[counter] = fig
    counter += 1    

    fig, (axa, axb) = plt.subplots(2, 1, figsize = (10.5, 6.25), dpi = 1200)
    axa.plot(nTime1, smoothed_data["TGS2602"].iloc[tRange1], linewidth = 0.55)
    axa.plot(nTime1, smoothed_data["BO TGS2602"].iloc[tRange1], linewidth = 0.55)
    axa.set_ylabel("Reading $[µV]$", fontsize = 13)
    axa.set_xlim(left=0, right = tRange)
    #axa.set_ylim(top = 800)
    axb.plot(nTime2, smoothed_data["TGS2602"].iloc[tRange2], linewidth = 0.55)
    axb.plot(nTime2, smoothed_data["BO TGS2602"].iloc[tRange2], linewidth = 0.55)
    axb.legend(["Ambient", "Box"], title = "Sensors")
    axb.set_xlabel(xLabel, fontsize = 13)
    axb.set_ylabel("Reading $[µV]$", fontsize = 13)
    axb.set_xlim(left=0, right = tRange)
    #axb.set_ylim(top = 800)
    axa.set_title(f"Boron {pcbNum} TGS2602", fontsize = 16, fontweight = 'bold')  

    figs[counter] = fig
    counter += 1
    
    fig, (axa, axb) = plt.subplots(2, 1, figsize = (10.5, 6.25), dpi = 1200)
    axa.plot(nTime1, smoothed_data["Temperature"].iloc[tRange1], linewidth = 0.55)
    axa.plot(nTime1, smoothed_data["BO Temperature"].iloc[tRange1], linewidth = 0.55)
    axa.set_ylabel("Reading $[µV]$", fontsize = 13)
    axa.set_xlim(left=0, right = tRange)
    #axa.set_ylim(top = 50)
    axb.plot(nTime2, smoothed_data["Temperature"].iloc[tRange2], linewidth = 0.55)
    axb.plot(nTime2, smoothed_data["BO Temperature"].iloc[tRange2], linewidth = 0.55)
    axb.legend(["Ambient", "Box"], title = "Sensors")
    axb.set_xlabel(xLabel, fontsize = 13)
    axb.set_ylabel("Reading $[µV]$", fontsize = 13)
    axb.set_xlim(left=0, right = tRange)
    #axb.set_ylim(top = 50)
    axa.set_title(f"Boron {pcbNum} Temperature $[°C]$", fontsize = 16, fontweight = 'bold')  

    figs[counter] = fig
    
    # Return plots for exporting
    return figs


#%% ------------------------------------------- Input date and range below -------------------------------------------  
figs0 = plotRange(startTime, 6)
"""
Plots data over a range of days given a starting date and number of days to plot.
----------
tStart : Starting date in yyyy-mm-dd format
tRange : Number of days to plot
-------
"""


#%% ------------------------------------------- Input dates and range below -------------------------------------------    
figs1 = compareRange("2025-07-02","2025-07-02", 6)
"""
Plots data over two ranges of days given each starting date and number of days to plot.
----------
t1Start : Starting date #1 in yyyy-mm-dd format
t2Start: Starting date #2 in yyyy-mm-dd format
tRange : Number of days to plot
-------
"""

#%% ------------------------------------------- Export Here -------------------------------------------
print("Exporting report...")

with PdfPages(f"Boron_{pcbNum}_SensorPlots.pdf") as pdf:
    for fig in figs0:
        fig += 1
        pdf.savefig(fig)
        plt.close(fig)

# If you want to export both figure sets...
# figs = {**figs0, **{max(figs0.keys()) + 1 +i: fig for i, fig in enumerate(figs1.values())}}
# del figs0, figs1
# with PdfPages(f"Boron_{pcbNum}_SensorPlots.pdf") as pdf:
#     for fig in figs.values():
#         pdf.savefig(fig)
#         plt.close(fig)