# -*- coding: utf-8 -*-
"""
@author: lulu

***********README*****************
example code:

import sys
path = 'C:/Users/lucie/OneDrive/Desktop/Boronius/Python/sheets_puller.py'
sys.path.append(path) #you can comment this and this ^^ out if you put sheets_puller.py in the same folder as your.py
from sheets_puller import list_dict as ld
from sheets_puller import list_list as ll
from sheets_puller import sheet_dataframe as sd

sheet = "Data00" #change sheet name. case sensitive (data00 wont work, Data00 will)

lis = ll(sheet) #creates a list of lists from the worksheet
dic = ld(sheet) #creates a list of dictionaries from the worksheet
df = sd(sheet)  #creates a dataframe from the worksheet
***********************************
"""

import gspread
import pandas as pd

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