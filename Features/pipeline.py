#Machine Learning for Public Policy: HW3
#Charity King
#PA3: Machine Learning Pipeline
#May 3, 2016
#csking1@uchicago.edu

import csv
import pandas as pd
import explore_clean as exp
import numpy as np
from sklearn import preprocessing, cross_validation, metrics, tree, decomposition, svm
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
import matplotlib.pyplot as plt
import time as t

MODELS = ['LR','DT', 'KNN', 'RF', 'NB', 'GB', 'AB', 'BG']

clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'BG': BaggingClassifier(n_estimators=10, n_jobs=-1)
            }

grid = {
    'RF':{'n_estimators': [1,10,100], 'max_depth': [1,5,10], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'ET': { 'n_estimators': [1,10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100]},
    'GB': {'n_estimators': [1,10,100], 'learning_rate' : [0.01,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,5,10]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,1]},
    'KNN' :{'n_neighbors': [1,5,10,25],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
    'BG': {'n_estimators': [1, 10, 100]}
           }

def magic_loop(dataframe, x, y):

    cls_d = {}

    with open('models.csv', 'w', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter=',')
        w.writerow(['MODEL', 'PARAMETERS', 'PRECISION', 'RECALL', 'AUC', 'F1', 'ACCURACY', 'Time'])
        with open('best_models.csv', 'w', newline='') as csvfile:
            c = csv.writer(csvfile, delimiter=',')
            c.writerow(['MODEL', 'PARAMETERS', 'PRECISION', 'RECALL', 'AUC', 'F1', 'ACCURACY', 'Time'])
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

            class_auc = {}
            best_model = ''
            best_AUC = 0
            best_param = ''
            best_ypred = ''
            best_precision = 0
            best_recall = 0
            best_f1 = 0
            best_accuracy = 0
            best_time = 0

            for index,clf in enumerate([clfs[x] for x in MODELS]):
        
                cls_ypred = ''
                current_model = MODELS[index]
                class_auc[current_model] = 0
                cls_ypred = ''
                print (current_model)
                parameter_values = grid[current_model]
                start_time = t.time()
                for p in ParameterGrid(parameter_values):
                    try:

                        start_time = t.time()
                        clf.set_params(**p)
                        y_pred_probs = clf.fit(x_train, y_train).predict_proba(x_test)[:,1]
                        precision, accuracy, recall, f1, threshold, AUC = model_evaluation(y_test, y_pred_probs,.05)
                        end_time = t.time()
                        total_time = end_time - start_time
                        w.writerow([current_model, p, precision, recall, AUC, f1, accuracy, total_time])
               
                        if AUC > class_auc[current_model]:
                            class_auc[current_model] = AUC
                            cls_ypred = y_pred_probs
                            cls_param = p
                            cls_precision = precision
                            cls_recall = recall
                            cls_f1 = f1
                            cls_accuracy = accuracy
                            cls_time = total_time
                   
                    except IndexError as e:
                        continue

                auc = class_auc[current_model]
                c.writerow([current_model, cls_param, cls_precision, cls_recall, auc, cls_f1, cls_accuracy, cls_time])
                cls_d[current_model] = [auc, cls_ypred]
                plot_precision_recall_n(y_test, cls_ypred, current_model)

def model_evaluation(y_true, y_scores, k):
    
    threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))]
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    precision = metrics.precision_score(y_true, y_pred)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    AUC = metrics.roc_auc_score(y_true, y_pred)

    return precision, accuracy, recall, f1, threshold, AUC


def plot_precision_recall_n(y_true, y_scores, model_name):
    '''
    Takes the model, plots precision and recall curves
    '''

    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_scores)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_scores)

    for value in pr_thresholds:
        num_above_thresh = len(y_scores[y_scores >= value])
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
    name = model_name
    plt.title(name)
    plt.savefig("Eval/{}.png".format(name))
    # plt.show()

def go(training, testing):

    labels = exp.read_data(testing)
    train = exp.read_data(training)
    exp.data_summary(train)
    exp.graph_data(train)
    exp.impute_data(train, mean=False, median=True)
    x, y = exp.feature_generation(train)
    magic_loop(train, x, y)

if __name__ == '__main__':
	training = "cs-training.csv"
	testing = "cs-test.csv"


go(training, testing)




 