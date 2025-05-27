# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 12:10:56 2025

@author: lucie
"""

import sys
#path = 'C:/Users/lucie/OneDrive/Desktop/Boronius/Python/sheets_puller.py'
#sys.path.append(path) #you can comment this and this ^^ out if you put sheets_puller.py in the same folder as your.py
from sheets_puller import list_dict as ld
from sheets_puller import list_list as ll
from sheets_puller import sheet_dataframe as sd

sheet = "DATA_FULL_SWEEP-4-14" #change sheet name. case sensitive (data00 wont work, Data00 will)

lis = ll(sheet) #creates a list of lists from the worksheet
dic = ld(sheet) #creates a list of dictionaries from the worksheet
df = sd(sheet)  #creates a dataframe from the worksheet


