import csv
import pandas as pd
import explore_clean as exp
import numpy as np

Y_VAR = "allegation_outcome"

def read_data(filename):
    df = pd.read_csv(filename)
    return df

def data_summary(dataframe):
 
    print("----------------Percentiles:-------------" "\n", np.round(dataframe.describe(percentiles = [.5]), 2).to_string(justify = "left"))
    print("----------------Mean:--------------------" "\n", dataframe.mean().to_string(float_format = "{:.2f}".format))
    print("----------------Median:------------------" "\n", dataframe.median().to_string(float_format = "{:.2f}".format))
    print("----------------Standard Deviation:------" "\n", dataframe.std().to_string(float_format = "{:.2f}".format))
    print("----------------Mode:--------------------" '\n', dataframe.mode().to_string(index = False))
    print("----------------Correlation Matrix:------" "\n", dataframe.corr())
    print("----------------Missing Values:----------" "\n", dataframe.isnull().sum().to_string())


def procurement_method(df):
    pass



def go(filename):
    df = read_data(filename)
    data_summary(df)
    # print (df.allegation_category)
    print (df.objective)

filename = "../Example/landing.csv"
go(filename)