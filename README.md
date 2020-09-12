# Multilayer Perceptron Implementation Using Spark And Python

## Dataset

The dataset is the [Customer Complaint Database](https://catalog.data.gov/dataset/consumer-complaint-database). It includes complains from customers on some financial products and services from 2011 untill today. The data file is a comma-delimited .csv file. 

### File Format


```
0 <- date %Y-%m-%d
1 <- category
2 <- comment
```

## Main Goal

The goal of this project is to use [Spark](https://spark.apache.org/) to implement a [Perceptron Classifier](https://en.wikipedia.org/wiki/Multilayer_perceptron) and [HDFS](https://hadoop.apache.org/docs/r1.2.1/hdfs_design.html) using the [TFIDF metric](https://en.wikipedia.org/wiki/Tf%E2%80%93idf). Both RDDs and Spark Dataframe API were utilized.

## Requirements
- nltk

## Usage

Assuming that Spark and HDFS are properly installed and working on our system:

- Upload data file in HDFS
```
hadoop fs -put ./customer_complaints.csv hdfs://master:9000/customer_complaints.csv
```

- Install necessary requirements
```
pip install -r requirements.txt
```

- Submit job in a Spark environment
```
spark-submit mlp.py
```


### Algorithm

- Clean the data
- Keep k most common words in all comments
- Remove less ofter categories
- Compute TFIDF metric for each word in the comments 
- Use SparseVector where key is word_index and value is tfidf metric
- Transform string labels (categories) in integers
- Split dataset in train and test set (stratified split)
- Train a Multilayer Perceptron model (fit)
- Compute accuracy of model on test set (transform)
