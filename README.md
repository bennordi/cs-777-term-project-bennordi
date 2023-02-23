# cs-777-term-project-bennordi
Term Project for BU MET CS 777 - Big Data Analytics

## Instructions
To run this project, you must first download the csv data from [Kaggle](https://www.kaggle.com/datasets/team-ai/spam-text-message-classification), and rename the file to `spam.csv` and move it to the directory of `noriega_termproject.py`.

From there, run the following command and the data will be pulled successfully and the script will run.
```bash
python3 noriega_termproject.py spam.csv
```

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
This is a simple classification dataset, which we will leverage Pyspark and various libraries to analyze text data to build various machine learning models that will predict `ham or spam` based on the given text data. We will then review each model and its results. There are 5572 rows of data to analyze, and luckily the data was preformatted so that `ham/spam` are the only values in the category column, and the message column will need to be formatted into features columns for us to build models off of.

This data will be split into training and test sets (70/30 split) and will be seeded for consistency of the split. The classification models will be trained on the same exact training data and will be tested with the same test dataset. Then during the results, I can do some anecdotal tests for messages I create to test out our models.

## Machine Learning Model(s)

## Results

## Conclusion

