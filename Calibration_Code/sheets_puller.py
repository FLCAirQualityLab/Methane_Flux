"""
Sheets_Puller
lulu & Jessica Goff
"""


# Importing libraries
import gspread
import pandas as pd


def sheetPuller(sheetName, spreadsheetName, credentials = 'credentials.json'):
    """
    Pulls a specific sheet from a Google Spreadsheet and returns it as a pandas DataFrame.

    Parameters:
    - sheetName (str): The name of the specific worksheet you want to pull (e.g., "Sheet1").
    - spreadsheetName (str): The name of the Google Spreadsheet file (e.g., "BoroniusData").
    - credentials (str): The file path to your Google Cloud service account JSON key.

    Returns:
    - pandas.DataFrame: A DataFrame containing the data from the worksheet.
    """
    # Authenticate with Google using the service account JSON file
    gc = gspread.service_account(filename = credentials)

    # Open the spreadsheet by its name
    spreadsheet = gc.open(spreadsheetName)

    # Select the specific worksheet by its name
    worksheet = spreadsheet.worksheet(sheetName)
    
    # Get all records from the worksheet and convert to a DataFrame
    dataframe = pd.DataFrame(worksheet.get_all_records())
    
    return dataframe

#%% Testing function...
# try:
#     # Call the function to get your data
#     my_data = sheetPuller(sheet_name="Data", spreadsheet_name="Boron_3")
    
#     # Print the first 5 rows of the DataFrame
#     print(my_data.head())

# except gspread.exceptions.SpreadsheetNotFound:
#     print("Error: The spreadsheet 'Boron_3' was not found. Please check the name and sharing settings.")
# except gspread.exceptions.WorksheetNotFound:
#     print("Error: The worksheet was not found. Please check the sheet name.")
# except FileNotFoundError:
#     print("Error: The credentials file was not found. Make sure 'credentials.json' is in the correct path.")
# except Exception as e:
#     print(f"An unexpected error occurred: {e}")