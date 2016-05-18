
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

TARGET = "Dlqin2yrs"
RENAME = ["Index", "Dlqin2yrs", "Unsecured", "Age", \
     "30to59days", "Debtratio", "Monthlyincome",  "Opencredit", \
    "90days","realestate", "60to89days", "Dependents"]

def read_data(filename):
    '''
    Takes raw filename and 2 CSV filenames
    Calls conditional function to create two csv files
    '''
    df = pd.read_csv(filename)
    df.columns = RENAME
    return df.drop(df.columns[[0]], axis = 1)

def data_summary(dataframe):
    pass

    print("----------------Percentiles:-------------" "\n", np.round(dataframe.describe(percentiles = [.5]), 2).to_string(justify = "left"))
    print("----------------Mean:--------------------" "\n", dataframe.mean().to_string(float_format = "{:.2f}".format))
    print("----------------Median:------------------" "\n", dataframe.median().to_string(float_format = "{:.2f}".format))
    print("----------------Standard Deviation:------" "\n", dataframe.std().to_string(float_format = "{:.2f}".format))
    print("----------------Mode:--------------------" '\n', dataframe.mode().to_string(index = False))
    print("----------------Correlation Matrix:------" "\n", dataframe.corr())
    print("----------------Missing Values:----------" "\n", dataframe.isnull().sum().to_string())

def graph_data(dataframe):
    dataframe.groupby(dataframe.columns[0]).size().plot(kind = "bar", width = 1, rot = 0)
    plt.show()
    for name in dataframe.columns[1:]:
        dataframe.groupby(name).size().plot()
        plt.show()
    ################Specialized graphs##################
    dataframe[dataframe.Debtratio < 3].Debtratio.hist(bins=2)
    plt.show()
   
def impute_data(dataframe, mean=False, median=False): 
    header = list(dataframe.columns)
    for each in header:
        if dataframe[each].isnull().values.any():
            if mean:
                dataframe[each] = dataframe[each].fillna(dataframe[each].mean())
    
            else:
                dataframe[each] = dataframe[each].fillna(dataframe[each].median())



def binary_variable(dataframe, variable):
    data[column] = data[column].apply(lambda x: 0 if \
        column == variable else 1)

def binning_data(dataframe, variable, bins):

    col = "bin " + str(variable)
    dataframe[col] = pd.cut(dataframe[variable], bins=bins, \
        include_lowest=True, labels=False)
    return col


def feature_generation(dataframe):
    y = dataframe[TARGET]
    y = np.ravel(y)
    x = dataframe.drop(TARGET,1)
    return x, y

def feature_importance(x, dataframe):

    features = x
    clf = RandomForestClassifier(compute_importances=True)
    clf.fit(dataframe[features], dataframe[TARGET])
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)
    padding = np.arange(len(features)) + 0.5
    pl.barh(padding, importances[sorted_idx], align='center')
    pl.yticks(padding, features[sorted_idx])
    pl.xlabel("Relative Importance")
    pl.title("Variable Importance")
    pl.show()