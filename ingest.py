import pandas as pd
import numpy as np
import requests
import StringIO as s



url = "https://finances.worldbank.org/api/views/kdui-wcs3/rows.csv?accessType=DOWNLOAD"
r = requests.get(url)
data = s.StringIO(r.content)
dataframe = pd.read_csv(data,header=0)
print (dataframe.head())

