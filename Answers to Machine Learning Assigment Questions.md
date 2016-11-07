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
 
 
 
 
 