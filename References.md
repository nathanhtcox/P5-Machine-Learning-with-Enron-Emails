### References

#### Numpy Primer
https://engineering.ucsb.edu/~shell/che210d/numpy.pdf
This primer was helpful because I wasn't really familiar with conditionally selecting and changing values in a numpy array. I think most of the numpy tricks I used didn't make it into my final code base (becasue I ended up modifying the 'my_dataset' dictionary directly) but it this primer was useful in the intermediate steps.

#### Udacity Forum Post
https://discussions.udacity.com/t/trying-to-hit-over-0-3/196167/14
This forum post gave me the useful suggestion that feature selection is highly critical so I went back and added a couple more features and this really boosted my results. I also got the idea to remove fewer outliers from this post. Originally I was removing outliers more than 3 IQR above the Q3 value. I increased this to 5 IQR above the Q3 value and this reduced the number of outliers I was removing. This also improved results as I was removing fewer POIs from the dataset.

#### sklearn Documentation
http://scikit-learn.org/stable/
Both the descriptions of the different classifiers and the user guides were helpful. The naive bayes, decision tree, SVC and random forest pages were used constantly as well as the SelectKBest and metrics library documentation.


#### Stack Overflow - General Python Questions
In addition to the references above, I used stack overflow quite a bit to sort out python programming issues as I haven't done a ton of python programming.