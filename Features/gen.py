import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

DROP_LIST = ['Unnamed: 0','Contract Signing Date','Total Contract Amount (USD)','Begin Appraisal Date',
       'Borrower Contract Number','Procurement Method ID', 'Project Name_y','approval_date',
       'bank_approval_date', 'begin_appraisal_date', 'begin_preparation_date',
       'closing_date','concept_review_date', 'contract_sign-off_date', 
       'date_case_opened', 'date_complaint_opened','decision_meeting_date', 'effectiveness_date',
       'no_objection_date','procurement_type_description', 'signing_date','boardapprovaldate', 'closingdate',
       'lendprojectcost', 'ibrdcommamt', 'idacommamt', 'totalamt', 'grantamt']

DUMMY_LIST = ['Region', 'Fiscal Year', 'Borrower Country','Borrower Country Code', 'Procurement Type', \
'Procurement Category','Procurement Method', 'Product line', 'Major Sector_x', 'Supplier Country', \
'Supplier Country Code', 'resolved_supplier', 'allegation_category',  'allegation_outcome', 'allegation_type', \
'complaint_status','country', 'lead_investigator', 'major_sector', 'procurement_method_id', 'regionname', 'vpu', 'prodline', 
'lendinginstr', 'lendinginstrtype','board_approval_month', 'borrower',  'impagency']

LOG_LIST = ['contract_amount']
BINARY_LIST = ['caseoutcome', 'project_amount'] 
BINNING_LIST = [('project_amount', 100)]
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
    return dataframe

def drop_columns(df):
    for col in DROP_LIST:
        df = df.drop(col, axis=1)
    return df

def get_log(df):
    '''
    get_log takes a string column, converts to a float, and then takes the log of values
    '''
    for col in LOG_LIST:
        df[col] = df[col].fillna(0)
        df[col] = df[col].apply(lambda x: np.log(x + 1))
    return df

def feature_generation(dataframe):
    '''
    Takes a df and separates the X columns from the y columns
    A simple LR  model is included to  ensure the X, Y columns can run through a model 
    '''
    y = dataframe[Y_VAR]
    y = np.ravel(y)
    x = dataframe.drop(Y_VAR, 1)
    # model = LogisticRegression()
    # model = model.fit(x, y)
    # testing = model.score(x, y)
    # print ("accuracy score of {}".format(testing))
    return x, y

def impute_zeros(df, column):
    '''
    Imputing function
    '''
    df[column] = df[column].fillna(0)

def drop_rows(df):
    df = df.dropna(subset = [Y_VAR])
    return df
def predictor_helper(x):
    '''
    Helper function to fix the predictor value
    '''
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

def feature_importance(x, y, k, df):
    '''
    Takes an x, y, top k value, and dataframe
    Returns a list of top k features as well as number of features that have 0 MDI and number of 
    features that have an MDI below .005. 
    Outputs feature distribution graph
    '''
    not_zero = 0
    under_01 = 0
    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
    forest.fit(x, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    for f in range(k):
        next = indices[f] + 1
        print("%d. feature %d, %s (%f)" % (f + 1, indices[f],  list(df.ix[:, indices[f]:next]), importances[indices[f]]))
    ######Reports how many variables have a 0 level 
    for f in range(x.shape[1]):
        if importances[indices[f]] != 0:
            not_zero +=1
        if importances[indices[f]] <= .005:
            under_01 +=1

    print ("Amount of features not zero is: %d" % (not_zero))
    print ("Amount of features under .0 is: %d" % (under_01))

    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(x.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
    plt.xticks(range(x.shape[1]), indices)
    plt.xlim([-1, x.shape[1]])
    plt.show()
def top_feature_analysis(df):
    '''
    A very ugly yet useful function for further exploring the best two performing features. Not general purpose 
    and not really good for anything else. 
    '''
    #######################top 1 feature################
    print (df['vpu'].value_counts())
    vpu = df.loc[df['caseoutcome'] == 1]
    print (vpu['vpu'].value_counts())
    #######################top 2 feature###############
    print (df['lead_investigator'].value_counts())
    invest = (len(list(df['lead_investigator'].value_counts())))
    invest = list(df['lead_investigator'].value_counts())
    plt.plot(invest)
    plt.xlabel("Number of Investigators")
    plt.ylabel("Number of Investigations")
    plt.show()
    baha = df.loc[df['caseoutcome'] == 1]
    print(baha['lead_investigator'].value_counts())

def go(filename):
    df = read_data(filename)
    fix_predictor(df, Y_VAR)
    top_feature_analysis(df)
    df = df.convert_objects(convert_numeric=True)
    df = get_dummies(df)
    create_binary(df)
    df = get_log(df)
    df = binning(df)
    df = drop_columns(df)
    x, y = feature_generation(df)
    feature_importance(x, y, 10, df)

    return x, y

if __name__ == "__main__":
    filename = '../Example/resolved_joined.csv'
go(filename)

