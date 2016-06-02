from __future__ import division
import pandas as pd
import numpy as np
from preprocess_data import update_with_cc_means
# from generate_features import cat_to_binary
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, recall_score, auc, f1_score
import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
import time
import csv


def define_clfs_params():

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3) 
            }

    grid = { 
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01]}, #,0.1,1,10
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]}, #,20,50,100
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }

    return clfs, grid

def magic_loop(models_to_run, clfs, params, X, y, k=0.5):
    '''
    '''
    rows = [['Models', 'Parameters', 'Split', 'AUROC', 'Accuracy at '+str(k), 'Recall at '+str(k), 'F1 at '+str(k), 'precision at ' + str(k)]]
    tracker=0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    for index,clf in enumerate([clfs[x] for x in models_to_run]):
        parameter_values = params[models_to_run[index]]
        for p in ParameterGrid(parameter_values):
            clf.set_params(**p)
            y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
            print(models_to_run[index] + ' ' + str(tracker) + '/1213')
            tracker += 1
            print(p)
            # print(len(y_test))
            # print(len(y_train))
            # print(y_train == y_test[1:])
            threshold = np.sort(y_pred_probs)[::-1][int(k*len(y_pred_probs))]
            y_pred = np.asarray([1 if i >= threshold else 0 for i in y_pred_probs])

            score_list = []
            score_list.append(metrics.accuracy_score(y_test, y_pred))
            score_list.append(metrics.recall_score(y_test, y_pred))
            score_list.append(metrics.f1_score(y_test, y_pred))
            score_list.append(metrics.precision_score(y_test, y_pred))
            print(score_list)
            # print(type(score_list))
            rows.append(score_list)
    return rows
            # print(type(y_pred_probs))
            # print(type(y_test))
            # print(y_pred_probs)
            # print(y_test)
            # for i in range(1, 20):
            #     try:
            #         print('real: {}; prob: {}'.format(y_test[i], y_pred_probs[i]))
            #     except:
            #         continue

# def magic_loop(models_to_run, clfs, params, X, y, Ks=[0.1, 0.5, 0.9]):
#     ''' 
#     X and y need to be formatted
#     '''
#     tracker = 0
#     model_list=[['Models', 'Parameters', 'Split', 'AUROC']]
#     for k in Ks:
#         model_list[0] += ['Accuracy at '+str(k), 'Recall at '+str(k), 'F1 at '+str(k), 'precision at ' + str(k)]
#     print(model_list)

#     for n in range(1, 2):
#         # print("split: {}".format(n))
        
#         for index,clf in enumerate([clfs[x] for x in models_to_run]):
#             # print(models_to_run[index])
            
#                 try:
#                     # d = {}
#                     # print("parameters {}".format(p))
#                     clf.set_params(**p)
#                     # clf.fit(X_train, y_train)
#                     y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]

#                     # y_pred_probs = clf.predict_proba(X_test)[:,1]
                    

#                     row=[models_to_run[index], p, n, roc_auc_score(y_test, y_pred_probs)]
#                     for k in Ks: #by default Ks has one value


#                         # print(y_pred_probs)
#                         # print()
#                         # for i in y_test:
#                         #     print(i)
#                         # print(y_test)

#                         row += score_list #evaluate_models_at_k(y_pred, X_test, y_test, k)
#                     # plot_precision_recall_n(y_test, y_pred_probs, clf)
#                     # model_list = [['Models', 'Parameters', 'Split', 'Accuracy at '+str(k), 'Recall at '+str(k), 'AUROC', 'F1 at '+str(k), 'precision at ' + str(k)]]
#                     # model_list.append( d['accuracy at '+str(k)], d['recall'], d['AUROC'], d['F1'], d['precision at ' + str(k)]])
#                     model_list.append(row)
#                     # print(pd.DataFrame(model_list))

#                     tracker += 1
#                     print(models_to_run[index] + ' ' + str(tracker) + '/1213')
#                     print(p)
#                 except IndexError as e:
#                     print('Error:',e)
#                     continue

#     return model_list


def get_summary():
    '''
    Takes output from magic loop and returns a summary of the highest values
    '''
    pass

def read_data(filename, response):
    '''
    Reads in from transformed csv file and generates X and Y arrays
    '''
    df = pd.read_csv(filename)
    ###############
    df = df.dropna()
    ###############
    Y = df[response]
    df = df.drop(response, 1)
    df = df.drop(df.columns[[0]], axis=1)
    return df, Y

def main(data_filename, response, output_filename, summary_filename): 
    models_to_run = models_to_run = ['LR', 'NB', 'DT', 'RF']
    clfs, params = define_clfs_params()
    X, y = read_data(data_filename, response)
    rows = magic_loop(models_to_run, clfs, params, X, y)
    # print(rows)
    for i in rows:
        print(i)
        print(type(i))


if __name__ == "__main__":
    main('cs-training.csv', 'SeriousDlqin2yrs', 'loop_full.csv', 'loop_summary.csv')