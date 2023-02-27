```bash
(base) ➜  cs-777-term-project-bennordi git:(main) ✗ python3 noriega_benjamin_term_project.py spam.csv
Starting Term Project

Time to load, preprocess and vectorize data: 25.456 secs                        

--------------------------------------------------
Logistic Regression

Accuracy: 0.978
Precision: 0.926
Recall: 0.910
F1: 0.918
Confusion Matrix: 
[[1431.   16.]
 [  20.  201.]]
Time to build, fit, and predict: 7.443 secs
--------------------------------------------------
Linear SVC

Accuracy: 0.980
Precision: 0.916
Recall: 0.932
F1: 0.924
Confusion Matrix: 
[[1428.   19.]
 [  15.  206.]]
Time to build, fit, and predict: 3.826 secs
--------------------------------------------------
Naive Bayes Classifier

Accuracy: 0.948                                                                 
Precision: 0.736
Recall: 0.946
F1: 0.828
Confusion Matrix: 
[[1372.   75.]
 [  12.  209.]]
Time to build, fit, and predict: 2.440 secs
--------------------------------------------------
Decision Tree Classifier

Accuracy: 0.947                                                                 
Precision: 0.908
Recall: 0.670
F1: 0.771
Confusion Matrix: 
[[1432.   15.]
 [  73.  148.]]
Time to build, fit, and predict: 5.666 secs
--------------------------------------------------
GBT Classifier

Accuracy: 0.960                                                                 
Precision: 0.914
Recall: 0.774
F1: 0.838
Confusion Matrix: 
[[1431.   16.]
 [  50.  171.]]
Time to build, fit, and predict: 14.998 secs
--------------------------------------------------
Random Forest Classifier

Accuracy: 0.871                                                                 
Precision: 1.000
Recall: 0.027
F1: 0.053
Confusion Matrix: 
[[1447.    0.]
 [ 215.    6.]]
Time to build, fit, and predict: 3.856 secs

--------------------------------------------------
Total Time: 63.687 secs
```