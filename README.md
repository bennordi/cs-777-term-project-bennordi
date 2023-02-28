# cs-777-term-project-bennordi
Term Project for BU MET CS 777 - Big Data Analytics

## Instructions
To run this project, clone it to your local machine which contains the script and `spam.csv` file containing our dataset.

From there, run the following command and the data will be pulled successfully and the script will run.
```bash
python3 noriega_termproject.py spam.csv
```

Make sure the spacy module is installed. In case you encounter an error regarding the `en_core_web_sm` model, run the following command:

```bash
python3 -m spacy download en_core_web_sm
```

Once this is installed, running the script command should run successfully.

## Dataset
CSV file found on Kaggle under [Spam Text Message Classification](https://www.kaggle.com/datasets/team-ai/spam-text-message-classification).

Format table:
`<category>`: string (ham/spam)
`<message>`: string (text)

| Category    | Message                      |
| ----------- | ---------------------------- |
| ham         | I'm gonna be home soon ...   |
| spam        | Free entry in 2 a weekly ... |

## Research Question
This is a simple classification dataset, which we will leverage PySpark and various libraries to analyze text data to build various machine learning models that will predict `ham or spam` based on the given text data. We will then review each model and its results. There are 5572 rows of data to analyze, and luckily the data was preformatted so that `ham/spam` are the only values in the category column, and the message column will need to be formatted into features columns for us to build models off of. The main need for preprocessing was removing non-letters from the text, lemmatizing the text, and then vectorizing the data (with MLlib’s pipeline using Tokenizer(), StopWordsRemover(), CountVectorizer(), IDF(). I was going to use ChiSqSelector() but we learned in Assignment 5 that with larger datasets this can be very computationally expensive, and knowing that the goal of this assignment is to be scalable, I opted to not perform this selection.

This data will be split into training and test sets (70/30 split) and will be seeded for consistency of the split. The classification models will be trained on the same exact training data and will be tested with the same test dataset. Then during the results, I can do some anecdotal tests for messages I create to test out our models.

## Machine Learning Models
Used 6 different machine learning models from PySpark's MLlib library, and fit the model with our training dataset, predicted on our test dataset. Output of my local machine run can be found on [output.md](output.md).

- Logistic Regression
- Linear SVC
- Naive Bayes classifier
- Decision Tree classifier
- Gradient-Boosted Tree classifier (GBT)
- Random Forest Classifier

## Conclusion
In terms of actual accuracy, Linear SVC and Logistic Regression were the most accurate at predicting the correct classification. In retrospect, based on the class distribution of ham/spam having the ‘spam’ class as positive makes the precision measure of this study a bit overrated. For example, at first look without seeing the confusion matrix, precision for the Random Forest model looks awesome at 100%, but to notice there were only 6 True Positives and 0 False Positives, the measure isn’t very valuable anymore. 

Knowing that there were 1447 negative (ham) items and 221 positive (spam) items in our test dataset, this study could have used more of a closer distribution between classes. Had the split in training/test datasets been more even in terms of class the results would have been much more interesting but it really comes down to the goal of the analysis and prediction of the models. Do we want to be better at predicting spam or correctly predicting ham in terms of text messages? Obviously, the majority of text messages are considered ‘ham’, or legitimate, so the distribution of this ‘sample’ data is likely a good representation of what real world data looks like, and I’d hazard to guess that the 13% spam occurrence is even larger than the actual amount of spam coming to people’s text messages.

The results were very accurate for each model, each being above 94% accurate except for the random forest classifier (87%), which was my most disappointing model, having a precision measure of 1 (for only 6 True positives and 0 False positives) and a Recall measure of 0.027. The overall strongest model in this study was logistic regression, having an accuracy of 97.8%, Precision measure of 0.926, Recall of 0.910, and an F1-score of 0.918.

An effort to expand this study could be performing different feature extraction, or going a different route utilizing PySpark’s NLP libraries to break down the raw data to then build classifier machine learning models. Certain transformers could have been used such as Chi-Square Selection but as mentioned above the computational expense would make this use case difficult if the dataset was much larger.
