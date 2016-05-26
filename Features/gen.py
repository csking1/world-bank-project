import csv
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression

DROP_LIST = ['Unnamed: 0', 'Unnamed: 0.1', 'as_of_date',
       'borrower_contract_reference_number',
       'borrower_country_code', 'contract_description',
       'contract_signing_date', 'country', 
       'project_id', 'project_name', 'region', 'supplier',
       'supplier_country', 'supplier_country_code', 'wb_contract_number',
       'resolved_supplier', 'orig_supplier_name', 'year_x', 'month',
       'Country Name_x', 'Country Code_x',
       'Indicator Name_x', 'Indicator Code_x', 'year_y',
       'Country Name_y', 'Country Code_y', 'Indicator Name_y',
       'Indicator Code_y','allegation_category',
       'outcome_val', 'wb_id', 'objective', 'competitive']

DUMMY_LIST = ['fiscal_year', 'major_sector', 'procurement_category', \
'procurement_method', 'procurement_type', 'product_line',  'country_name_standardized', \
'ppp']

BINARY = ['allegation_outcome', ]

Y_VAR = "allegation_outcome"

def read_data(filename):
    '''
    Takes in csv filename and outputs pandas df
    '''
    df = pd.read_csv(filename)
    return df

def cat_to_binary(df):

    for col in DUMMY_LIST:
        dummies = pd.get_dummies(df[col], col, drop_first=True)
        DROP_LIST.append(col)
        df = df.join(dummies)
    return df

def binary_helper(x):

    if x == "Unsubstantiated":
        return int(0)
    if x == "Substantiated":
        return int(1)
    else:
        return None

def create_binary(df, column):
    '''
    Takes a df and column wtih categorical binary vaues
    Transforms into numberical binary values 0 and 1
    '''

    df[column] = df[column].apply(binary_helper)

def binning_data(dataframe, variable, bins):

    col = "bin " + str(variable)
    dataframe[col] = pd.cut(dataframe[variable], bins=bins, \
        include_lowest=True, labels=False)
    return col

def drop_columns(df):
    for col in DROP_LIST:
        df = df.drop(col, axis=1)
    return df

def feature_generation(dataframe):
    y = dataframe[Y_VAR]
    y = np.ravel(y)
    x = dataframe.drop(Y_VAR, 1)
    model = LogisticRegression()
    model = model.fit(x, y)
    testing = model.score(x, y)
    print ("accuracy score of {}".format(testing))
    return x, y
def impute_zeros(df, column):
    df[column] = df[column].fillna(0)

def go(filename):
    df = read_data(filename)
    df = cat_to_binary(df)
    df = drop_columns(df)
    create_binary(df, Y_VAR)
    impute_zeros(df, Y_VAR)
    x, y = feature_generation(df)
    return x, y

if __name__ == '__main__':
    filename = "../Example/landing.csv"
x, y = go(filename)
