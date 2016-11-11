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
    my_dataset = pickle.load(data_file)

### Task 2: Remove outliers
### By visual inspection I have noticed that "Total" and "The Travel Agency in the Park" appear to be outliers that are not relevant
### to the investigation
my_dataset.pop("TOTAL")
my_dataset.pop("THE TRAVEL AGENCY IN THE PARK")

### Task 3: Create new feature(s)
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

### Selecting the 10 best features 
sel = SelectKBest(f_classif, k=10)

select_features = sel.fit_transform(features, labels)
select_features = np.array(select_features)

### creating the output 'features_list' that the tester routine needs to operate
features_list = ['poi']
for num in sel.get_support(True):
    features_list.append(all_features[num+1])

print features_list
print sel.get_support(True)
print sel.scores_


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

    
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

clf_nb = GaussianNB()
clf_dtree = DecisionTreeClassifier()
clf_svc = SVC()
clf_forest = RandomForestClassifier()
clf_ada = AdaBoostClassifier(n_estimators=25, learning_rate=2)


for clf in [clf_nb, clf_dtree, clf_svc, clf_forest, clf_ada]:
    test_classifier(clf, my_dataset, features_list)
#test_classifier(clf_nb, my_dataset, features_list)

    
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.grid_search import GridSearchCV

#tuning the parameters on decision tree
#param_grid = {'criterion':['gini','entropy'], 'splitter':['best','random'], 'max_depth':[5,10,15,20,50],'min_samples_split':[2,3,4,5,6,7]}

#tuning the parameters on ada boost
param_grid = {'n_estimators':[5,25,50,75,100], 'learning_rate':[1,2,3,4]}

ada = AdaBoostClassifier()
clf = GridSearchCV(ada, param_grid, 'f1')
clf.fit(select_features, labels)
print clf.best_params_


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf_ada, my_dataset, features_list)
