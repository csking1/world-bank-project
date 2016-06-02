import csv
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression

DROP_LIST = ['Unnamed: 0',
       'Contract Signing Date',
       'Total Contract Amount (USD)',
       'Begin Appraisal Date', 'Borrower Contract Number',
       'Procurement Method ID', 'Project Name_y', 'allegation_category',
       'allegation_outcome', 'allegation_type', 'approval_date',
       'bank_approval_date', 'begin_appraisal_date', 'begin_preparation_date',
       'closing_date', 'complaint_status',
       'concept_review_date', 'contract_amount', 'contract_sign-off_date',
       'country', 'date_case_opened', 'date_complaint_opened',
       'decision_meeting_date', 'effectiveness_date', 'lead_investigator',
       'major_sector', 'no_objection_date', 'procurement_method_id',
       'procurement_type_description', 'project_amount', 'signing_date', 'vpu',
       'regionname', 'prodline', 'lendinginstr', 'lendinginstrtype',
       'boardapprovaldate', 'board_approval_month', 'closingdate',
       'lendprojectcost', 'ibrdcommamt', 'idacommamt', 'totalamt', 'grantamt',
       'borrower', 'impagency']

DUMMY_LIST = ['Region', 'Fiscal Year', 'Borrower Country','Borrower Country Code', 'Procurement Type', \
'Procurement Category','Procurement Method', 'Product line', 'Major Sector_x', 'Supplier Country', \
'Supplier Country Code', 'resolved_supplier']
# DUMMY_LIST = ['Region', 'Fiscal Year', 'major_sector', 'procurement_category', \
# 'procurement_method', 'procurement_type', 'product_line',  'country', 'country_name_standardized', \
# 'supplier_country_code']
BINARY_LIST = ['caseoutcome']
# BINARY_LIST = ['allegation_outcome', 'objective', 'competitive']

BINNING_LIST = [("amount_standardized", 10), ("ppp", 20)]

Y_VAR = 'caseoutcome'

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
def predictor_helper(x):
    if x == "Substantiated":
        return int(1)
    else:
        return int(0)

def fix_predictor(df, Y_VAR):
    '''
    Takes a df and column wtih categorical binary vaues
    Transforms into numberical binary values 0 and 1
    '''
    df[Y_VAR] = df[Y_VAR].apply(predictor_helper)

def go(filename):
    df = read_data(filename)
    fix_predictor(df, Y_VAR)
    # binning(df)
    df = get_dummies(df)
    create_binary(df)

    df = drop_columns(df)
    x, y = feature_generation(df)
    return x, y

# if __name__ == "__main__":
#     filename = '../Example/resolved_joined.csv'
# go(filename)