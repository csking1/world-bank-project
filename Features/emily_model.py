from __future__ import division
import pandas as pd
import numpy as np
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
import random
#import pylab as pl
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy import optimize
import time
from sklearn.metrics import precision_recall_curve
import sklearn.metrics as metrics
import datetime
import heapq
import gen as gen

# change this to take in as a parameter
FILE = '/Users/Emily/Desktop/Harris/ML 101/Assignments/ML-Programing-Assignments/PA-02/Output'

def define_clfs_params():
    '''
    Initializes dictionaries of classifiers and parameters
    '''
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
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }
    return clfs, grid

def magic_loop(models_to_run, clfs, grid, X, y):
    '''
    Takes a list of models to use, two dictionaries of classifiers and parameters, and array of X
    '''
    table = {}
    top = []
    for i in range(10):
        top.append((0, " "))
    heapq.heapify(top)
    k = 0.05
    for n in range(1, 2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        for index, clf in enumerate([clfs[x] for x in models_to_run]):
            for p in ParameterGrid(grid[models_to_run[index]]):
                try:
                    clf.set_params(**p)
                    # print (clf)
                    y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                    plot_precision_recall_n(y_test, y_pred_probs, clf)
                    l = scoring(k, y_test, y_pred_probs)
                    m, s = top[0]
                    auc = l['auc']
                    if auc > m:
                        print ("switching out {}".format(auc))
                        heapq.heapreplace(top, (auc, clf))

                except:
                    print ('Error:')
                    continue
    return top

def scoring(k, y_test, y_pred_probs):
    '''
    Takes results of classifier, adds metrics to result table,
    '''
    l = {}
    l['precision'], y_scores = precision_at_k(y_test, y_pred_probs, k)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_probs)
    l['fpr'] = fpr
    l['tpr'] = tpr
    l['auc'] = metrics.auc(fpr, tpr)
    # try:

    #     l['accuracy'] = metrics.accuracy_score(y_test, y_scores)
    # except:
    #     print ("Couldn't get result metrics here")

    return l

def plot_precision_recall_n(y_true, y_prob, model_name):
    '''
    Takes the model, plots precision and recall curves
    '''
    # why the copy here? They both reference the same thing.
    y_score = y_prob

    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)

    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score >= value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)

    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')

    name = str(model_name)
    try:
        plt.title(name)
        plt.savefig("Output/Images/{}.png".format(name))
    except:
        name = name[:15]
        plt.title(name)
        plt.savefig("Output/Images/{}.png".format(name))
    plt.close()

def precision_at_k(y_true, y_scores, k):
    '''
    For a given level of K, return the precision score
    '''
    threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))]
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    return metrics.precision_score(y_true, y_pred), y_scores

def record_table(table):
    '''
    Takes dictionary, prints out results to a file
    '''
    with open("Output/Final_table.txt", 'w') as f:
        f.write("Top Ten AUC Classifiers \n")
        for clf, auc in table:
            f.write("\nAUC: {} from {}\n".format(str(clf), auc))
    return

def get_x_and_y(filename):
    df = pd.read_csv(filename)
    Y = df['SeriousDlqin2yrs']
    df = df.drop('SeriousDlqin2yrs', 1)
    df = df.drop(df.columns[[0]], axis=1)
    return df, Y

def main(filename):
    '''
    Executes functions sequentially, records main classifiers to output text file
    '''
    clfs, grid = define_clfs_params()
    # models_to_run = ['LR','ET','AB','GB','NB','DT', 'KNN','RF']
    models_to_run = ['NB']
    # X, y = get_x_and_y(filename)
    X, y = gen.go('../Example/resolved_joined.csv')
    top =  magic_loop(models_to_run, clfs, grid, X, y)
    record_table(top)

if __name__ == '__main__':
    print ("================= Running test at {} ====================".format(str(datetime.datetime.now())))
    main(FILE)
