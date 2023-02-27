# Ben Noriega
# CS777
# 02.22.23
# Term Project

import sys
import time
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.classification import LogisticRegression, LinearSVC, NaiveBayes, GBTClassifier, DecisionTreeClassifier, RandomForestClassifier
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.feature import CountVectorizer, IDF, StopWordsRemover, Tokenizer
from pyspark.ml import Pipeline

import spacy
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# Load the spaCy model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def vectorize(df):
	"""
	Converts our dataframe into vectorized text
	
	Params:
		df - DataFrame - represents our dataset

	Returns:
		DataFrame - vectorized dataframe
	"""
	tokenizer = Tokenizer(inputCol="Message", outputCol="words")
	remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="filtered")
	countVectorizer = CountVectorizer(inputCol=remover.getOutputCol(), outputCol="features_c", vocabSize=5000)
	idf = IDF(inputCol=countVectorizer.getOutputCol(), outputCol="features")
	pipeline = Pipeline(stages=[tokenizer, remover, countVectorizer, idf])

	model = pipeline.fit(df)

	return model.transform(df)

# Taken from course Github page - Spark-Example-23-Mllib-Sentiment%20Model.ipynb
def lemmatize_text(text):
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    return " ".join(lemmas)

def get_metrics(cm):
	"""
	Calculates and prints out performance metrics for the predictions made.
	Taken from my assignment 5 submission.

	Params:
		cm - ndarray - represents confusion matrix

	Returns:
		N/A - prints out results
		
	"""
	tp = cm[0][0]
	fp = cm[0][1]
	fn = cm[1][0]
	tn = cm[1][1]

	try:
		accuracy = (tp + tn) / (tp + fp + fn + tn)
	except:
		accuracy = 0
	try:
		prec = tp / (tp + fp)
	except:
		prec = 0
	try:
		rec = tp / (tp + fn)
	except:
		rec = 0
	try:
		f1 = 2 * (prec * rec) / (prec + rec)
	except: 
		f1 = 0
	
	print(f"Accuracy: {accuracy:.3f}")
	print(f"Precision: {prec:.3f}")
	print(f"Recall: {rec:.3f}")
	print(f"F1: {f1:.3f}")
	print(f"Confusion Matrix: \n{cm}")
	return

def logistic_regression(trainData, testData):
	"""
	Builds a logistic regression model based on our training data and predicts based on our classifier.

	Params:
		trainData - DataFrame - training portion of our dataset
		testData  - DataFrame - test portion of our dataset

	Returns:
		N/A - result metrics are printed to the console
	"""
	print("-"*50)
	print("Logistic Regression")
	print()

	lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20)
	model = lr.fit(trainData)

	predictions = model.transform(testData)
	predsLabels = predictions.select("prediction", "label").rdd
	metrics = MulticlassMetrics(predsLabels)
	get_metrics(metrics.confusionMatrix().toArray())


def linear_svc(trainData, testData):
	"""
	Builds a linear SVC model based on our training data and predicts based on our classifier.

	Params:
		trainData - DataFrame - training portion of our dataset
		testData  - DataFrame - test portion of our dataset

	Returns:
		N/A - result metrics are printed to the console
	"""
	print("-"*50)
	print("Linear SVC")
	print()

	lsvc = LinearSVC(featuresCol="features", labelCol="label", maxIter=20)
	model = lsvc.fit(trainData)

	predictions = model.transform(testData)
	predsLabels = predictions.select("prediction", "label").rdd
	metrics = MulticlassMetrics(predsLabels)
	get_metrics(metrics.confusionMatrix().toArray())

def decision_tree_classifier(trainData, testData):
	"""
	Builds a decision tree classifier model based on our training data and predicts based on our classifier.

	Params:
		trainData - DataFrame - training portion of our dataset
		testData  - DataFrame - test portion of our dataset

	Returns:
		N/A - result metrics are printed to the console
	"""
	print("-"*50)
	print("Decision Tree Classifier")
	print()

	dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")
	model = dt.fit(trainData)

	predictions = model.transform(testData)
	predsLabels = predictions.select("prediction", "label").rdd
	metrics = MulticlassMetrics(predsLabels)
	get_metrics(metrics.confusionMatrix().toArray())

def naive_bayes_classifier(trainData, testData):
	"""
	Builds a Naive Bayes classifier model based on our training data and predicts based on our classifier.

	Params:
		trainData - DataFrame - training portion of our dataset
		testData  - DataFrame - test portion of our dataset

	Returns:
		N/A - result metrics are printed to the console
	"""
	print("-"*50)
	print("Naive Bayes Classifier")
	print()

	nb = NaiveBayes(featuresCol="features", labelCol="label")
	model = nb.fit(trainData)

	predictions = model.transform(testData)
	predsLabels = predictions.select("prediction", "label").rdd
	metrics = MulticlassMetrics(predsLabels)
	get_metrics(metrics.confusionMatrix().toArray())

def gbt_classifier(trainData, testData):
	"""
	Builds a Gradient-Boosted tree classifier model based on our training data and predicts based on our classifier.

	Params:
		trainData - DataFrame - training portion of our dataset
		testData  - DataFrame - test portion of our dataset

	Returns:
		N/A - result metrics are printed to the console
	"""
	print("-"*50)
	print("GBT Classifier")
	print()
	gbt = GBTClassifier(maxIter=20, featuresCol="features", labelCol="label", maxDepth=5)
	model = gbt.fit(trainData)

	predictions = model.transform(testData)
	predsLabels = predictions.select("prediction", "label").rdd
	metrics = MulticlassMetrics(predsLabels)
	get_metrics(metrics.confusionMatrix().toArray())

def random_forest_classifier(trainData, testData):
	"""
	Builds a Random Forest classifier model based on our training data and predicts based on our classifier.

	Params:
		trainData - DataFrame - training portion of our dataset
		testData  - DataFrame - test portion of our dataset

	Returns:
		N/A - result metrics are printed to the console
	"""
	print("-"*50)
	print("Random Forest Classifier")
	print()

	rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=10)
	model = rf.fit(trainData)

	predictions = model.transform(testData)
	predsLabels = predictions.select("prediction", "label").rdd
	metrics = MulticlassMetrics(predsLabels)
	get_metrics(metrics.confusionMatrix().toArray())


if __name__ == "__main__":
	print("Starting Term Project")
	print()

	sc = SparkContext(appName="Term Project")
	spark = SparkSession.builder.getOrCreate()

	################################################################
	## Stage 1: Preparing Data
	t = time.time()

	df = spark.read.csv(sys.argv[1], inferSchema=True, header=True)
	# Creating label column
	df = df.withColumn("label", F.when(F.col("Category") == "spam", 1.0).otherwise(0.0))

	# Remove non-letters
	df = df.withColumn("Message", F.regexp_replace("Message", r"[^a-zA-Z ]", ""))

	# Define a UDF to apply the lemmatizer to a column
	lemmatize_udf = udf(lemmatize_text, StringType())

	# Applying UDF
	df = df.withColumn("Message", lemmatize_udf(df["Message"])).cache()

	df = vectorize(df)
	
	# Splitting training and test data
	trainData, testData = df.randomSplit(weights=[0.7, 0.3], seed=7335)

	print(f"Time to load, preprocess and vectorize data: {time.time() - t:.3f} secs")
	print()

	################################################################
	## Stage 2: Building Machine Learning Models - Evaluation

	# Models are built and predicted in each helper method - results printed to console
	tm = time.time()
	logistic_regression(trainData, testData)
	print(f"Time to build, fit, and predict: {time.time() - tm:.3f} secs")

	tm = time.time()
	linear_svc(trainData, testData)
	print(f"Time to build, fit, and predict: {time.time() - tm:.3f} secs")

	tm = time.time()
	naive_bayes_classifier(trainData, testData)
	print(f"Time to build, fit, and predict: {time.time() - tm:.3f} secs")

	tm = time.time()
	decision_tree_classifier(trainData, testData)
	print(f"Time to build, fit, and predict: {time.time() - tm:.3f} secs")
	
	tm = time.time()
	gbt_classifier(trainData, testData)
	print(f"Time to build, fit, and predict: {time.time() - tm:.3f} secs")
	
	tm = time.time()
	random_forest_classifier(trainData, testData)
	print(f"Time to build, fit, and predict: {time.time() - tm:.3f} secs")

	print()
	print("-"*50)
	print(f"Total Time: {time.time() - t:.3f} secs")
