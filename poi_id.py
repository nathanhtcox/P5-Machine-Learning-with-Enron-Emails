#!/usr/bin/python

import sys
import pickle
import pprint
import numpy as np
import matplotlib.pyplot
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import MinMaxScaler

### Task 1: Select what features you'll use.
### I will use all features at the start and add additional features then run the complete feature list through SelectKBest to determine
### which features to keep. 

all_features = ['poi','salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'from_this_person_to_poi', 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Store to my_dataset for easy export below
my_dataset = data_dict

### Create new features and add them to my_dataset
### My guess is that ratios may be more powerful than absolute numbers, so I've created a few new features that are logical ratios.
for name, data in my_dataset.iteritems():
    data['perc_email_from_poi']=np.nan_to_num(float(data['from_poi_to_this_person'])/float(data['to_messages']))
    data['perc_email_to_poi']=np.nan_to_num(float(data['from_this_person_to_poi'])/float(data['from_messages']))
    data['perc_totalinc_from_exercstock'] = np.nan_to_num(float(data['exercised_stock_options'])/float(data['total_payments']))
    
all_features.append('perc_email_from_poi')
all_features.append('perc_email_to_poi')
all_features.append('perc_totalinc_from_exercstock')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, all_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

features=np.array(features)
labels=np.array(labels)

### Selecting the 5 best features 
sel = SelectKBest(f_classif, k=5)

select_features = sel.fit_transform(features, labels)
select_features = np.array(select_features)

### creating the output 'features_list' that the tester routine needs to operate
features_list = ['poi']
for num in sel.get_support(True):
    features_list.append(all_features[num+1])


print features_list
print sel.get_support(True)
print sel.scores_

############################
### Task 2: Remove outliers
############################
### I'm using 5 x interquartile range for each selected to delete outliers. I feel this is less arbitrary than selecting outliers
### by visual inspection.


#get Q1, Q3 and IQR
q1 = np.percentile(select_features, 25, axis=0)
q3 = np.percentile(select_features, 75, axis=0)
iqr = q3 - q1 

to_delete = []
i=0
for feature in features_list:
    if feature == 'poi':
        continue

    for k,person in my_dataset.iteritems():
        if float(person[feature]) >=  (q3[i] + 5 * iqr[i]):
            to_delete.append(k)
    i=i+1

for k in set(to_delete):
    my_dataset.pop(k, None)


### Task 3: Create new feature(s)
### I created the new features before running SelectKBest to see if the features I created were better than others
### I removed outliers after creating and selecting features because removing outliers for for features that are unimportant may
### needlessly reduce the dataset.


### scaling features so that SVM will work properly
min_max_scaler = MinMaxScaler()
scaled_features = min_max_scaler.fit_transform(select_features)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.cross_validation import train_test_split, cross_val_score
features_train, features_test, labels_train, labels_test = \
    train_test_split(scaled_features, labels, test_size=0.3, random_state=42)
    
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

clf_nb = GaussianNB()
clf_svc = SVC(C=10000) #setting C very high to see if some POIs can be classified
clf_dtree = DecisionTreeClassifier()
clf_forest = RandomForestClassifier()

for clf in [clf_nb, clf_svc, clf_dtree, clf_forest]:
    test_classifier(clf, my_dataset, features_list)

    
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.grid_search import GridSearchCV

#tuning the parameters on decision tree
param_grid = {'criterion':['gini','entropy'], 'splitter':['best','random'], 'max_depth':[5,10,15,20,50],'min_samples_split':[2,3,4,5,6,7]}

dtree = DecisionTreeClassifier()
clf = GridSearchCV(dtree, param_grid, 'f1')
clf.fit(select_features, labels)
print clf.best_params

### GridSearchCV seemed to return a different set of 'ideal' parameters each time
### these are the parameters that seem fairly stable each time GridSearchCV is run, max_depth appears to change each time
clf_dtree = DecisionTreeClassifier(criterion='gini', splitter='random', min_samples_split=3)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf_dtree, my_dataset, features_list)