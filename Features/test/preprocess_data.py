#Functions to find/replace missing data in various ways


import pandas as pd
from numpy.random import choice
from sklearn.cross_validation import train_test_split 
import numpy as np

def update_with_mean(data):
    '''
    Updates the dataframe with the overall mean as instructed in 3A
    '''
    columns = data.columns.values
    for col in columns:
        if type(col.ix[3]) == str or type(col.ix[3]) == bool:
            continue
        filler = data[col].mean()
        data[col] = data[col].fillna(filler)
    return data

def update_with_cc_means(data, groupvar):
    '''
    Updates the dataframe with the class-conditional mean as instructed in 3B
    '''
    columns = data.columns.values
    means = data.groupby(groupvar).mean() # syntax like a dictionary: means[att][y/n]
    for col in columns:
        for i, b in enumerate(data[col].isnull()):
            if b:
                att = data[groupvar][i]
                val = means[col][att]
                data.set_value(i, col, val)
    return data

def update_with_variance(data, columns):
    '''
    Because all the data are whole numbers, I find the probability of a randomly chosen person to be
    a given value, and assign missing values based on that.

    Implemented as instructed in 3C.

    For categorical variable.

    '''
    vc = {}
    for col in columns:
        vc[col] = data[col].value_counts()
    for col in columns:
        vals = vc[col].axes[0]
        probs = list(map(lambda x: x/sum(vc[col]), vc[col]))
        data[col].fillna(int(choice(vals, 1, probs)))
    return data 


def split_train_test(data_x, data_y, prop_in_test):
    '''
    splits and formats for logit function
    '''
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=float(prop_in_test))
    return X_train, X_test, y_train, y_test

