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