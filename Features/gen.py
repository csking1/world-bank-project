import csv
import pandas as pd
import explore_clean as exp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression

DROP = []
BINARY = []
CATEG = []
LOG = []
#variables to drop:
#as_of_date
#use feature selection as a preliminary
#
Y_VAR = "allegation_outcome"

def read_data(filename):
    df = pd.read_csv(filename)
    return df

def binary_variable(dataframe, variable):
    data[column] = data[column].apply(lambda x: 0 if \
        column == variable else 1)


def binary(dataframe, column):
    dummies = pd.get_dummies(df[column])
    df = df.join(dummies)
    return df

def binning_data(dataframe, variable, bins):

    col = "bin " + str(variable)
    dataframe[col] = pd.cut(dataframe[variable], bins=bins, \
        include_lowest=True, labels=False)
    return col

def feature_generation(dataframe):
    y = dataframe[Y_VAR]
    y = np.ravel(y)
    x = dataframe.drop(Y_VAR, 1)
    model = LogisticRegression()
    model = model.fit(x, y)
    testing = model.score(x, y)
    print ("accuracy score of {}".format(testing))

    # return x, y


def go(filename):
    df = read_data(filename)
    feature_generation(df)


 
    # print (df.as_of_date)
    # print (df.borrower_contract_reference_number)
    # print (df.country)
    # print (df.country, df.borrower_country_code)
    # print (df.contract_description)
    # print (df.contract_signing_date)
    # print (df.major_sector)
    # print (df.procurement_category)
    # print (df.columns)





if __name__ == '__main__':
    filename = "../Example/landing.csv"
go(filename)





# def feature_importance(x, dataframe):

#     features = x
#     clf = RandomForestClassifier(compute_importances=True)
#     clf.fit(dataframe[features], dataframe[TARGET])
#     importances = clf.feature_importances_
#     sorted_idx = np.argsort(importances)
#     padding = np.arange(len(features)) + 0.5
#     pl.barh(padding, importances[sorted_idx], align='center')
#     pl.yticks(padding, features[sorted_idx])
#     pl.xlabel("Relative Importance")
#     pl.title("Variable Importance")
#     pl.show()
