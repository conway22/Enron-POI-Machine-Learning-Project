#!/usr/bin/python

# Import functions and set the environment

import sys
import pickle
import matplotlib.pyplot as plt
import math
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features

payments = ['salary',
           'bonus',
           'long_term_incentive',
           'deferred_income',
           'deferral_payments',
           'loan_advances',
           'other',
           'expenses',
           'director_fees',
           'total_payments',]

stock_value = ['exercised_stock_options',
               'restricted_stock',
               'restricted_stock_deferred',
               'total_stock_value']

email_features = ['email_address',
                  'from_messages',
                  'from_poi_to_this_person',
                  'from_this_person_to_poi',
                  'shared_receipt_with_poi',
                  'to_messages']

feature_list = ['poi'] + payments + stock_value + email_features

### Load the dictionary containing the dataset

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Various functions used throughout the project

# Function to count values for feature, optional setting POI
def get_count(key, is_poi = None):
    count = 0
    for k, v in data_dict.items():
        if is_poi is not None:
            if v[key] != 'NaN' and v['poi'] == is_poi:
                count += 1
        else:
            if v[key] != 'NaN':
                count += 1
    return count

# Function to Compute Ratios
def get_ratio(numerator, denominator):

    ### in case of numerator or denominator having "NaN" value, return 0.
    ratio = round(float(numerator)/float(denominator), 2)
    ratio = ratio if not math.isnan(ratio) else 0
    return ratio

# Function that converts 'NaN' string to np.nan returning a pandas
# dataframe of each feature and it's corresponding percent null values
def get_nan_counts(dictionary):
    
    my_df = pd.DataFrame(dictionary).transpose()
    nan_counts_dict = {}
    for column in my_df.columns:
        my_df[column] = my_df[column].replace('NaN',np.nan)
        nan_counts = my_df[column].isnull().sum()
        nan_counts_dict[column] = round(float(nan_counts)/float(len(my_df[column])) * 100,1)
    df = pd.DataFrame(nan_counts_dict,index = ['percent_nan']).transpose()
    df.reset_index(level=0,inplace=True)
    df = df.rename(columns = {'index':'feature'})
    return df

# Function takes a dict, 2 strings, and shows a 2d plot of 2 features


def get_plot2d(data_set, feature_x, feature_y, title):
    
    data = featureFormat(data_set, [feature_x, feature_y])
    for point in data:
        x = point[0]
        y = point[1]
        plt.scatter( x, y )
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(title)
    plt.show()

    ### Identify who the Outliers are for a particular feature
    # for feature on the y-axis
    identity = []
    for person in data_dict:
        if data_dict[person][feature_y] != "NaN":
            identity.append((person, data_dict[person][feature_y]))
    print("%s outliers:" % feature_y)
    identity.sort(key = lambda x: x[1], reverse=True)
    print(identity[:5])
    
    # for feature on the x-axis
    identity = []
    for person in data_dict:
        if data_dict[person][feature_x] != "NaN":
            identity.append((person, data_dict[person][feature_x]))
    print("%s outliers:" % feature_x)
    identity.sort(key = lambda x: x[1], reverse=True)
    print(identity[:5])

#########################################################################
#  Exploratory Data Analysis (EDA)
#########################################################################

### Counts of total data point and features

# Total number of data points

print("There are %i people in the dataset" %len(data_dict))
print("There are %i features used per person" %(len(feature_list)))
print("Total number of POI's in the dataset: %i" % get_count('poi', True))
print("Total number of non-POI's in the dataset: %i" % get_count('poi', False))
print "\n",
print("The total number of data points in the dataset is %i" %(len(data_dict) * len(feature_list)))

total_non_nans = 0
for feature in feature_list:
    total_non_nans = total_non_nans + (get_count(feature, True) + get_count(feature, False))
print("The total number of non-NaNs in the dataset is %i with percent of total = %.2f"\
      %(total_non_nans, get_ratio(total_non_nans, (len(data_dict) * len(feature_list)))))


# Are there features with many missing values? etc.

print "\n",
print("Legend - feature|Total non-NaNs|%|non_Nans for POIs|%|non_Nans for non-POIs|%)")
print "\n",
    
for feature in feature_list:
        print(feature,
              get_count(feature, True) + get_count(feature, False),
              get_ratio((get_count(feature, True) + get_count(feature, False)), len(data_dict)),
              get_count(feature, True),
              get_ratio(get_count(feature, True), get_count('poi', True)),
              get_count(feature, False),
              get_ratio(get_count(feature, False), get_count('poi', False))
             )

#########################################################################
# Remove Outliers
##########################################################################

my_dataset = data_dict

# The plot before removing outliers
get_plot2d(data_dict, 'total_stock_value', 'total_payments', 'All Data')

### Task 2: Remove outliers
my_dataset.pop("TOTAL", 0) # Remove outlier predetermined from mini-project
my_dataset.pop('LOCKHART EUGENE E', 0) # has no data
my_dataset.pop('THE TRAVEL AGENCY IN THE PARK', 0) # Obviously travel agency has nothing to do with Enron

# The plot after removing outliers
get_plot2d(my_dataset, 'total_stock_value', 'total_payments', 'No Outliers')

### Other plots of interest:
#get_plot2d(data_dict, 'total_payments', 'total_stock_value', 'All Data')
#get_plot2d(data_dict, 'salary', 'other', 'All Data')
#get_plot2d(data_dict, 'salary', 'bonus', 'All Data')
#get_plot2d(data_dict, 'salary', 'expenses', 'All Data')
#get_plot2d(data_dict, 'salary', 'exercised_stock_options', 'All Data')

#########################################################################
###  Task 3: Create new feature(s)
###  Store to my_dataset for easy export below.#
#########################################################################

#### New features I would like to add: 
#### ratio of from_poi_to_this_person to to_messages,
#### ratio of from_this_person_to_poi to from_messages

for person in my_dataset:
    
    msg_from_poi = my_dataset[person]['from_poi_to_this_person']
    all_to_msg = my_dataset[person]['to_messages']
    if msg_from_poi != "NaN" and all_to_msg != "NaN":
        my_dataset[person]['msg_from_poi_ratio'] = get_ratio(msg_from_poi, all_to_msg)
    else:
        my_dataset[person]['msg_from_poi_ratio'] = 0
        
    msg_to_poi = my_dataset[person]['from_this_person_to_poi']
    all_from_msg = my_dataset[person]['from_messages']
    if msg_to_poi != "NaN" and all_from_msg != "NaN":
        my_dataset[person]['msg_to_poi_ratio'] = get_ratio(msg_to_poi, all_from_msg)
    else:
        my_dataset[person]['msg_to_poi_ratio'] = 0


### reset the features here so I can adjust as necessary
# grouped features

payments = ['salary',
           'bonus',
           'long_term_incentive',
           'deferred_income',
           'deferral_payments',
           'loan_advances',
           'other',
           'expenses',
           'director_fees',
           'total_payments',]

stock_value = ['exercised_stock_options',
               'restricted_stock',
               'restricted_stock_deferred',
               'total_stock_value']

email_features = [#'email_address',
                  'from_messages',
                  'from_poi_to_this_person',
                  'from_this_person_to_poi',
                  'shared_receipt_with_poi',
                  'to_messages']

new_features = ['msg_to_poi_ratio',
                'msg_from_poi_ratio']

### Feature list adjustments

features_list = ['poi'] + payments + stock_value# + email_features + new_features
#features_list = ['poi'] + stock_value + email_features

# Rerun the non-NaNs table with the chosen features_list

print '\n',("Legend - feature|Total non-NaNs|%|non_Nans for POIs|%|non_Nans for non-POIs|%)"),'\n'
    
for feature in features_list:
        print(feature,
              get_count(feature, True) + get_count(feature, False),
              get_ratio((get_count(feature, True) + get_count(feature, False)), len(data_dict)),
              get_count(feature, True),
              get_ratio(get_count(feature, True), get_count('poi', True)),
              get_count(feature, False),
              get_ratio(get_count(feature, False), get_count('poi', False))
             )
        
print '\n',

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#########################################################################
#  Feature Selection using SelectKBest
#########################################################################

### call the sklearn APIs (version 0.17.1)
from sklearn.feature_selection import chi2, f_classif, SelectKBest
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy as np
from pprint import pprint
from time import time
from numpy import mean



### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Select the best features using SelectKBest
k = 10
k_best = SelectKBest(k=k)
k_best.fit(features, labels)
scores = k_best.scores_
pairs = zip(features_list[1:], scores)
#combined scores and features into a pandas dataframe then sort 
k_best_features = pd.DataFrame(pairs,columns = ['feature','score'])
k_best_features = k_best_features.sort_values('score',ascending = False)
    
#merge with null counts    
df_nan_counts = get_nan_counts(my_dataset)
k_best_features = pd.merge(k_best_features,df_nan_counts,on= 'feature')  
    
#eliminate infinite values
k_best_features = k_best_features[np.isinf(k_best_features.score)==False]

print 'Feature Selection by SelectKBest\n'
#print "{0} best features in descending order: {1}\n".format(k, k_best_features.feature.values[:k])
print '{0}\n'.format(k_best_features[:k])

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#########################################################################
#  Variety of Classifiers: GaussianNB Classifier
#########################################################################

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# set up pipeline and parameter grid for GridSearchCV

param_grid = {'select__k': [8, 10, 'all'],
              'decomp__n_components': [1, 2, 4]
             }


pipe = Pipeline(steps=[('scaler', StandardScaler()),
                      ('select', SelectKBest()),
                      ('decomp', PCA()), 
                      ('classifier', GaussianNB())
                     ])

# then:

# create instance of StratifiedShuffleSplit (in this case with folds, you can change that)
folds = 1000
sss = StratifiedShuffleSplit(labels, folds, test_size=0.1, random_state=60)

# use StratifiedShuffleSplit as the cross-validation method for GridSearchCV
# i.e. cv=sss
grid_search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=sss, scoring='f1', verbose=1)

# As you are using StratifiedShuffleSplit, fit all of the data
# because GridSearchCV will create the test/train splits automatically
# when StratifiedShuffleSplit is the cross-validation method
grid_search.fit(features, labels)

# print the results:
print 'Best score: %0.3f' % grid_search.best_score_
print 'Best parameters set:'
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
       print '\t%s: %r' % (param_name, best_parameters[param_name])



# the optimal model from GridSearchCV is:
clf = grid_search.best_estimator_

print '\n', clf

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

'''

#########################################################################
#  Variety of Classifiers: DecisionTreeClassifier
#########################################################################

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# set up pipeline and parameter grid for GridSearchCV

param_grid = {"classifier__criterion": ["gini", "entropy"],
              "classifier__min_samples_split": [2, 10],
              "classifier__max_depth": [None, 2, 5],
              "classifier__min_samples_leaf": [1, 5],
              "classifier__max_leaf_nodes": [None, 5],
              'select__k': [10, 'all'],
              'decomp__n_components': [2, 10]
              }



pipe = Pipeline(steps=[('scaler', StandardScaler()),
                      ('select', SelectKBest()),
                      ('decomp', PCA()), 
                      ('classifier', DecisionTreeClassifier())
                     ])


# then:

# create instance of StratifiedShuffleSplit (in this case with 100 folds, you can change that)
folds = 1000
sss = StratifiedShuffleSplit(labels, folds, test_size=0.1, random_state=60)

# use StratifiedShuffleSplit as the cross-validation method for GridSearchCV
# i.e. cv=sss
grid_search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=sss, scoring='f1', verbose=1)

# As you are using StratifiedShuffleSplit, fit all of the data
# because GridSearchCV will create the test/train splits automatically
# when StratifiedShuffleSplit is the cross-validation method
grid_search.fit(features, labels)

# print the results:
print 'Best score: %0.3f' % grid_search.best_score_
print 'Best parameters set:'
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
       print '\t%s: %r' % (param_name, best_parameters[param_name])

# the optimal model from GridSearchCV is:
clf = grid_search.best_estimator_

print '\n', clf

dump_classifier_and_data(clf, my_dataset, features_list)

#########################################################################
#  Variety of Classifiers: AdaboostClassifier
#########################################################################

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# set up pipeline and parameter grid for GridSearchCV

param_grid = {"adaboost__n_estimators": [5, 10, 100],
              "adaboost__learning_rate": [0.1, 0.5, 1, 2],
              "adaboost__algorithm": ['SAMME', 'SAMME.R'],
              'decomp__n_components': [2, 10]
             }


pipe = Pipeline(steps=[('decomp', PCA()),
                       ('adaboost', AdaBoostClassifier())
                      ])


# then:

# create instance of StratifiedShuffleSplit (in this case with 100 folds, you can change that)
folds = 1000
sss = StratifiedShuffleSplit(labels, folds, test_size=0.1, random_state=60)

# use StratifiedShuffleSplit as the cross-validation method for GridSearchCV
# i.e. cv=sss
grid_search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=sss, scoring='f1', verbose=1)

# As you are using StratifiedShuffleSplit, fit all of the data
# because GridSearchCV will create the test/train splits automatically
# when StratifiedShuffleSplit is the cross-validation method
grid_search.fit(features, labels)

# print the results:
print 'Best score: %0.3f' % grid_search.best_score_
print 'Best parameters set:'
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
       print '\t%s: %r' % (param_name, best_parameters[param_name])

# the optimal model from GridSearchCV is:
clf = grid_search.best_estimator_

print '\n', clf

dump_classifier_and_data(clf, my_dataset, features_list)

'''