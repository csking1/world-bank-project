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
       'contract_signing_date', 
       'project_id', 'project_name', 'supplier',
       'supplier_country', 'wb_contract_number',
       'resolved_supplier', 'orig_supplier_name', 'year_x', 'month',
       'Country Name_x', 'Country Code_x',
       'Indicator Name_x', 'Indicator Code_x', 'year_y',
       'Country Name_y', 'Country Code_y', 'Indicator Name_y',
       'Indicator Code_y','allegation_category',
       'outcome_val', 'wb_id']

DUMMY_LIST = ['region','fiscal_year', 'major_sector', 'procurement_category', \
'procurement_method', 'procurement_type', 'product_line',  'country', 'country_name_standardized', \
'supplier_country_code']

BINARY_LIST = ['allegation_outcome', 'objective', 'competitive']

BINNING_LIST = [("amount_standardized", 10), ("ppp", 20)]

Y_VAR = "allegation_outcome"

def read_data(filename):
    '''
    Takes in csv filename and outputs pandas df
    '''
    df = pd.read_csv(filename)
    return df

def get_dummies(df):

    for col in DUMMY_LIST:
        dummies = pd.get_dummies(df[col], col)
        DROP_LIST.append(col)
        df = df.join(dummies)
    return df

def create_binary(df):
    '''
    Takes a df and column wtih categorical binary vaues
    Transforms into numberical binary values 0 and 1
    '''
    for column in BINARY_LIST:
        df[column] = df[column].fillna("missing")
        df[column] = df[column].astype('category')
        df[column] = df[column].cat.codes
    # df[column] = df[column].apply(binary_helper)

def binning_helper(dataframe, variable, bins):

    col = "bin " + str(variable)
    dataframe[col] = pd.cut(dataframe[variable], bins=bins, \
        include_lowest=True, labels=False)
    return col

def binning(dataframe):

    for each in BINNING_LIST:
        variable = each[0]
        bins = each[1]
        col = binning_helper(dataframe, variable, bins) 
        dataframe[each] = dataframe[col]

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

def drop_rows(df):
    df = df.dropna(subset = [Y_VAR])
    return df

def go(filename):
    df = read_data(filename)
    # print (df.columns)

    df = drop_rows(df)
    df = get_dummies(df)
    df = drop_columns(df)
    create_binary(df)
    binning(df)
    x, y = feature_generation(df)
    return x, y

if __name__ == "__main__":

    filename = '../Example/landing.csv'
go(filename)