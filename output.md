```bash
(base) ➜  cs-777-term-project-bennordi git:(main) ✗ python3 noriega_benjamin_term_project.py spam.csv
Starting Term Project

Time to load, preprocess and vectorize data: 26.433 secs                        

--------------------------------------------------
Logistic Regression

Accuracy: 0.978                                                                 
Precision: 0.989
Recall: 0.986
F1: 0.988
Confusion Matrix: 
[[1431.   16.]
 [  20.  201.]]
Time to build, fit, and predict: 7.879 secs
--------------------------------------------------
Linear SVC

Accuracy: 0.980
Precision: 0.987
Recall: 0.990
F1: 0.988
Confusion Matrix: 
[[1428.   19.]
 [  15.  206.]]
Time to build, fit, and predict: 3.963 secs
--------------------------------------------------
Naive Bayes Classifier

Accuracy: 0.948                                                                 
Precision: 0.948
Recall: 0.991
F1: 0.969
Confusion Matrix: 
[[1372.   75.]
 [  12.  209.]]
Time to build, fit, and predict: 2.631 secs
--------------------------------------------------
Decision Tree Classifier

Accuracy: 0.947                                                                 
Precision: 0.990
Recall: 0.951
F1: 0.970
Confusion Matrix: 
[[1432.   15.]
 [  73.  148.]]
Time to build, fit, and predict: 6.258 secs
--------------------------------------------------
GBT Classifier

Accuracy: 0.960                                                                 
Precision: 0.989
Recall: 0.966
F1: 0.977
Confusion Matrix: 
[[1431.   16.]
 [  50.  171.]]
Time to build, fit, and predict: 15.095 secs
--------------------------------------------------
Random Forest Classifier

Accuracy: 0.884                                                                 
Precision: 1.000
Recall: 0.882
F1: 0.937
Confusion Matrix: 
[[1447.    0.]
 [ 193.   28.]]
Time to build, fit, and predict: 3.701 secs

--------------------------------------------------
Total Time: 65.961 secs
```
