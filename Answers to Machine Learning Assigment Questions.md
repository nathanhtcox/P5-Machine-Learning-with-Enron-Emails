### Answers to Questions for Udacity Project 5 Machine Learning

Nathan Cox

#### Question 1
The goal of the project is to determine which Enron employees might be worth investigating for fraud based on data collected from Enron corporate emails and financial records. Machine learning is useful because it can create a model that can predict if an employee should be investigated by finding correlations in a large set of data. To create a model, the first step is to study the dataset and create new features  that may improve the predictive capability of the model. Then all features are tested for their predictive value and the best features are retained. Next the feature set is cleaned by removing outliers. Then different algorithms are tested to see which has the best predictive results. Finally the most promising algorithm is tuned to improve results further. 

The dataset contains information on salaries, stock options, emails sent and received and other financial and email related data from approximately 150 employees that worked at Enron. This dataset also includes a field that indicates if the person is a "Person of Interest" (POI). This field was created by hand and contains the labels that the algorithms will use to learn to search for POIs. There are 2 obvious outliers in the dataset based on visual inspection that should be removed. "Total" is simply the sum of all the other data points and does not correspond to an actual person. "The Travel Agency in the Park" also appears to not correpsond to an actual person and was removed. Even though there are some data points that had values significantly higher than the mean, they were not removed because some of them were POIs and in this case we believe all data points to be legitimate. There is no reason to remove them as this removes information.

##### Questions from the Rubric
There are 146 data points in the original dataset, of which 2 are outliers that were removed.
There are 18 POIs in the dataset, and 126 non-POIs (excluding outliers). This a dataset where the class allocation is skewed so it is important to use performance metric other than accuracy (a high accuracy score could be attained just by guessing all individuals are not a POI).
SelectKBest was used with the score parameter set to f_classif and k set to 10. f_classif is selected because the Annova F  is a good single measure that considers both recall and precision and in this case the goal is to maximize both.In the final model, ten features were selected. This was determined by iteratively trying different numbers of features from 1 to 14. Ten features gives the highest value for recall and the third highes value for precision. This seems like the point that gives the best tradeoff between the two perfrmance metrics.
![](PerformancevNumFeatures.png)
Some featu res have a large number of blank values and other features have values for most entries. Total Payments, Exercised Stock Options, Restricted Stock and Total Stock Value have numbers for most people. Loan Advances, Director Fees and Restricted Stock Deferred do not have many entries. Features that have values for most members are more likely to have predictive value than those with few entries.


#### Question 2
The features that were used and their scores from SelectKBest are below:
 - 18.3 - salary
 - 20.8 - bonus
 - 9.21 - restriced_stock
 - 11.5 - deferred_income
 - 24.2 - total_stock_value
 - 24.8 - exercised_stock_options
 - 9.92 - long_term_incentive
 - 9.21 - restricted_stock
 - 16.4 - perc_email_to_poi
 - 9.29 - perc_totalinc_from_exercstock
 
 The feature selection process is discussed in the rubric questions part of section 1.
 
 A SVM classifier was tested and feature scaling was implemented to do so, however the SVM algorithm gave very poor results (identified no POIs) and was removed from the code and not considered further. This is why feature scaling does not appear in the final code. Naive Bayes and Decision Trees do not require feature scaling to work properly (Random Forest and Ada Boost both rely on Decision Trees at their core).
 
 Three new features were created from the original dataset. Two of these features made it into the final model as they turned out to have a high predictive value. Ratios were used to bring out some interesting trends. Instead of just looking at the total number of emails sent/received to/from a POI, it is better to look at the ratio of the number of emails sent/recevied to/from a POI to the *total* number of emails sent/received. This will differentiate between those having a high percentage of their emails going to POIs and those who just happen to send a large number of emails. This similar ratio approach was used to break down the financial numbers as well by taking the ratio of the number of stock options exercised to the total payments that an employee received. 
 
#### Question 3 
The final algorithm selected was Gaussian Naive Bayes and this gave a precision of 49.3% and a recall of  40.5%. AdaBoost based on Decision Trees, Decision Tree, Random Forest and SVM classifiers were also tested. Gaussian Naive Bayes performed the even after AdaBoost and Decision Trees went through parameter optimzation. The performance of the other three algorithms are below

| Algorithm | Precision (%) | Recall (%) |
|:--------------:|:---------------:|:------------:|
|Decision Tree | 26 | 26|
|Random Forest | 41 | 14 |
|**Gaussian Naive Bayes** | **49** | **41**|
|AdaBoost | 30 | 20 |

SVM was particularly difficult to get a result with this dataset. It continually failed to identify POIs. Even with feature selection, scaling, removal of outliers and parameter tuning, it did not provide good results.

#### Question 4
Parameter tuning means selecting the parameters that will give the highest desired performance metric. In some cases performance metrics have certain trade-offs. For example it may be possible to get a higher precision score by sacrificing recall and vice versa. Performance tuning allows the selection of parameters to obtain the right trade off in performance metrics. The Ada Boost algorithm was tuned by using GridSearchCV to iterate over many possible combinations for 2 key parameters (n_estimators and learning_rate) to determine which set of parameters gives the highest f score. F score is chosen as the scoring metric because it weighs both recall and precision.

#### Question 5
 Validation is the process of ensuring that the model is generalizable to data outside of the specific data set that was used to train the algorithm. To do this, the dataset is split into training data and testing data. The training data is used to build/train the model. The testing data is used purely to verify the algorithm will work on more generalized data. The classic mistake that this avoids is overfitting the model to the specific data set. Overfitting means that the model will perform well on the training data set but not on any other datasets thereby limiting the model's predictive capabilities. The models were tested by using the code in the tester.py file that was provided. This uses stratified shuffle split to randomly separate the dataset into testing and training sets and does this several times in a random fashion to make the best use of the limited number of data points in the dataset.
 
#### Question 6
The 2 performance metrics used were precision and recall. Precision is the the likelihood that a classification is correct given the algorithm classified a datapoint. In this case it means the likelihood that person actually is a POI if the model predicted it. Recall is the likelihood that the model will correctly classify a person as a POI if they are actually a POI.
 