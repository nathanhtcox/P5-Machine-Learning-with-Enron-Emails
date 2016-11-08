### Answers to Questions for Udacity Project 5 Machine Learning

written by: Nathan Cox

#### Question 1
The goal of the project is to determine which Enron employees might be worth investigating for fraud based on data collected from Enron corporate emails and financial records. Machine learning is useful because it can create a model that can predict employees if an employee should be investigated based on a large set of data. To create a model, the first step is to study the dataset and create new features  that may improve the predictive capability of the model. Then all features are tested for their predictive value and the best features are retained. Next the feature set is cleaned by removing outliers. The final step is to try different algorithms and see which has the best predictive results and then tune that algorithm to improve results further. 

The dataset contains information on salaries, stock options, emails sent and received and other financial and email related data from approximately 150 employees that worked at Enron. This dataset also includes information on if the person is a "Person of Interest" (POI). These are the labels that we will use to train the algorithm to search for POIs. There are some outliers in the dataset that should be removed as the dataset has some erroneous data points. For example, the data point "total" is simply the sum of all the values of the other data points and does not correspond to an actual person. This value should be removed. I used a methodical approach to removing outliers. I calculated the interquartile range (IQR) and 25th and 75th percentile (Q1 and Q3) for each of the features that were selected. A data was considered an outlier and removed if it was greater than Q3 + 5 * IQR. Data points on the low end were not removed as Q1 for all features was 0 because there were several employees that did not have data for all features.

#### Question 2
The features that were used and their scores from SelectKBest are below:
 - 5.45 - from_poi_to_this_person
 - 8.90 - shared_receipt_with_poi
 - 3.29 - perc_email_from_poi
 - 16.9 - perc_email_to_poi
 - 9.46 - perc_totalinc_from_exercstock 
 
 I used SelectKBest with the score parameter set to f_classif and k set to 5. f_classif is selected because the f measure is a good single measure that considers both recall and precision and in this case I am trying to maximize both. I determined 5 features was best in the end by interatively trying more and less features and I found that 5 features gave the best combination of precision and recall.
 
 I did try to get a SVM classifier working and I implemented feature scaling to do it, however I was not able to get good results so I scrapped that algorithm and focussed on the other 4. This is why feature scaling does not appear in the final code. Naive Bayes and Decision Trees do not require feature scaling to work properly (Random Forest and Ada Boost both rely on Decision Trees at their core).
 
 I created 3 new features from the original dataset. All 3 of these features made it into my final model as they turned out to have a high predictive value. I used ratios to bring out some interesting trends. Instead of just looking at the total number of emails sent/received to/from a POI, it is better to look at the ratio of the number of emails sent/recevied to/from a POI to the *total* number of emails sent/received. This will differentiate between those communicating with POIs almost exclusively and those who just happen to send a huge number of emails. This similar ratio approach was used to break down the financial numbers as well by taking the ratio of the number of stock options exercised to the total payments that an employee received. 
 
#### Question 3 
I ended up using Ada Boost algorithm based on Decision Trees. I also tried Gaussian Naive Bayes, Decision Tree, Random Forest and SVM classifier. Ada Boost performed the best under default parameters at a precision of about 49% and a recall of 39%. The results for the next 3 are below:

| Algorithm | Precision (%) | Recall (%) |
|:--------------:|:---------------:|:------------:|
|Decision Tree | 38 | 38|
|Random Forest | 40 | 20 |
|Gaussian Naive Bayes | 27 | 34 |

SVM was particularly difficult to get a result with this dataset. It continually failed to identify POIs. Even with feature selection, scaling, removal of outliers and parameter tuning, it did not provide good results.

#### Question 4
Parameter tuning means selecting the parameters that will give the highest desired performance metric. In some cases performance metrics have certain trade-offs. For example it may be possible to get a higher precision score by sacrificing recall and vice versa. Performance tuning allows the selection of parameters to select the right trade off in performance metrics. The Ada Boost algorithm was tuned by selecting 2 of the critical parameters and using GridSearchCV to iterate over many possible parameter combinations to determine which set of parameters gives the highest f score. F score is chosen as the scoring metric because it weighs both recall and precision.

#### Question 5
 Validation is the process of ensuring that the model is generalizable to data outside of the specific data that was used to test the algorithm. To do this, the dataset is split into training data and testing data. The training data is used to build/train the model. The testing data is used purely to verify the algorithm will work on more generalized data. The classic mistake that this avoids is overfitting the model to the specific data set. Overfitting means that the model will perform well on the training data set but not on any other datasets thereby limiting the model's predictive capabilities. I tested the models by using the code in the tester.py file that was provided. This uses stratified shuffle split to randomly separate the dataset into testing and training sets and does this several times in a random fashion to make the best use of the limited number of data points in the dataset.
 
#### Question 6
The 2 performance metrics used were precision and recall. Precision is the the likelihood that a classification is correct given the algorithm classified a datapoint. In this case it means the likelihood that person actually is a POI if the model predicted it. Recall is the likelihood that the model will correctly classify a person as a POI if they are actually a POI.
 
 