import csv
import pandas as pd
import explore_clean as exp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#variables to drop:
#as_of_date
#use feature selection as a preliminary
#
Y_VAR = "allegation_outcome"

def read_data(filename):
    df = pd.read_csv(filename)
    return df

def data_summary(dataframe):
    pass
 
    # print("----------------Percentiles:-------------" "\n", np.round(dataframe.describe(percentiles = [.5]), 2).to_string(justify = "left"))
    # print("----------------Mean:--------------------" "\n", dataframe.mean().to_string(float_format = "{:.2f}".format))
    # print("----------------Median:------------------" "\n", dataframe.median().to_string(float_format = "{:.2f}".format))
    # print("----------------Standard Deviation:------" "\n", dataframe.std().to_string(float_format = "{:.2f}".format))
    # print("----------------Mode:--------------------" '\n', dataframe.mode().to_string(index = False))
    # print("----------------Correlation Matrix:------" "\n", dataframe.corr())
    # print("----------------Missing Values:----------" "\n", dataframe.isnull().sum().to_string())


def procurement_method(df):
    pass



def go(filename):
    df = read_data(filename)
    data_summary(df)
    # print (df.as_of_date)
    # print (df.borrower_contract_reference_number)
    # print (df.country)
    # print (df.country, df.borrower_country_code)
    # print (df.contract_description)
    # print (df.contract_signing_date)
    # print (df.major_sector)
    print (df.procurement_category)
    print (df.columns)





if __name__ == '__main__':
    filename = "../Example/landing.csv"
go(filename)