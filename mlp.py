from pyspark.sql import SparkSession
import time
from pyspark.sql.functions import date_format
from pyspark.sql.functions import lit
from pyspark.ml.linalg import SparseVector
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import re
import math

### customer_complaints.csv format:
"""
0 <- date %Y-%m-%d
1 <- category
2 <- comment
"""

eng_stopwords = set(stopwords.words('english'))
def clear_comment(text):
	"""
	Clears the comments: Removes punctuation, strange characters, digits, empty strings, stopwords,
	and keeps unique words in each comment

	Params:
	text (string): the comment from a customer

	Returns:
	text (tuple): Cleared comment
	"""

	# Split based on words, remove punctuation/strange characters
	cleared_text = re.compile(r'\W+', re.UNICODE).split(text.lower())
	# Remove digits ???
	text_without_digits = [re.sub('[\d_]', '', t) for t in cleared_text]
	# Remove empty strings from list of text
	text_without_empty_strings = [t for t in text_without_digits if t != '']
	# Remove stopwords
	text_without_stopwords = [word for word in text_without_empty_strings if word not in eng_stopwords]
	# Remove words that start with 'xx'
	text = [word for word in text_without_stopwords if not word.startswith('xx')]
	return tuple(text)


def get_tf_metric(text):
	"""
	Computes the tf metric

	Params:
	text (tuple): tuple of words

	Returns:
	tf_text: format: ((word1, word2, ...), (tf1, tf2, ...))
	"""

	counts = [text.count(word) for word in text]
	max_count = max(counts)
	tf = [counts[i]/max_count for i in range(0, len(counts))]

	return text, tf

def get_unique_words_in_each_text(text):
	"""
	Keeps unique words in each text

	Params:
	text (tuple): tuple of words

	Returns:
	unique_text
	"""

	unique_text = []
	text = list(text)
	[unique_text.append(x) for x in text if x not in unique_text]
	return tuple(unique_text)



if __name__ == "__main__":
	# Create Spark Context
	spark = SparkSession.builder.appName('questions_part1_new').getOrCreate()
	sc = spark.sparkContext

	# Load data
	data = sc.textFile("hdfs://master:9000/customer_complaints.csv")
	# Filter the data: keep lines that have 3 columns
	data = data.filter(lambda x: len(x.split(',')) == 3)
	# Filter the data: keep strings that start with '201'
	filtered_data = data.map(lambda x: (x.split(','))). \
			filter(lambda x: str(x[0]).startswith('201'))
	# Filter the data again: keep categories and comments that are not empty
	filtered_again_data = filtered_data.filter(lambda x: str(x[1]) != '' and str(x[2]) != "")
	# Clear the comments, not using date since it is not needed for later
	data_with_clear_comments = filtered_data.map(lambda x: (x[1], clear_comment(x[2])))
	# Create unique ID for each document -> output format: (doc_id, (category, comment))
	data_with_id = data_with_clear_comments.zipWithIndex().map(lambda x: (x[1], x[0]))
	data = data_with_id.filter(lambda x: len(x[1][1]) != 0).cache() #cache needed??????


	# Get words in all documents with k most common words
	#na dokimasw me kai xwris distinct, epishs tha bgalw apo edw to index, kai na dokimasw uinique words apo kathe document h oxi
	k = 1000 #20000
	words = data.flatMap(lambda x: [(y,1) for y in get_unique_words_in_each_text(x[1][1])]). \
		reduceByKey(lambda x, y: x + y). \
		sortBy(lambda x : x[1], ascending = False). \
		zipWithIndex(). \
		filter(lambda x: x[1] < k). \
		map(lambda x: (x[0][0], (x[0][1], x[1])) ). \
		cache() # output format: (word, (word_count, word_index))

	# Get vocabulary
	vocab = words.map(lambda x: x[0]). \
		collect() #take()

	#print('\n\n\n ', vocab[0:20])
	print('\n\n\nVocabulary size:', len(vocab))

	# Broadcast vocabulary to all executors
	broad_vocab = sc.broadcast(vocab)

	# Keep words of documents that exist in vocabulary, remove empty documents -> output format: (doc_id, (category, comment))
	data = data.map(lambda x: (x[0], (x[1][0], [word for word in x[1][1] if word in broad_vocab.value])) ). \
		filter(lambda x: len(x[1][1]) > 2).filter(lambda x: x[0] < 200000) # != 0

	# Check less often categories
	category_counts = data.map(lambda x: (x[1][0], 1)). \
		reduceByKey(lambda x, y: x+y). \
		sortBy(lambda x : x[1], ascending = False). \
		collect()
	print('\n\n', category_counts)
	# Remove less often categories
	data = data.filter(lambda x: x[1][0] != 'Other financial service' and x[1][0] != 'Virtual currency') #291, 16 comments on these categories only

	# Get number of documents
	N = data.count()
	print('\n\n\nNumber of documents: ', N, '\n\n\n')

	# Compute idf metric -> output format: (word, (idf, word_index))
	word_idf = words.map(lambda x: (x[0], (math.log(N/x[1][0]), x[1][1])))

	# Remove from memory
	words.unpersist()


	### Compute tf metric -> output format: (word, category, docId), (1, comment)))
	word_tf = data.flatMap(lambda x: [((word, x[0], x[1][0]), (1, x[1][1])) for word in x[1][1]])
	# output format: (word, category, docId), word counts, comment))
	word_tf = word_tf.reduceByKey(lambda x, y : (x[0] + y[0], x[1]) )
	# output format: (word, (category, docId, tf))
	word_tf = word_tf.map(lambda x: (x[0][0], (x[0][1], x[0][2], x[1][0]/len(x[1][1]))) ).cache()
	# output format: (word, (category, docId, tf))
	#word_tf = word_tf.map(lambda x: (x[0][0], (x[0][1], x[0][2] , x[1])) ).cache()

	#print('\n\n\n ', word_tf.count(), '\n\n\n')

	### Compute tfidf metric -> output format: (word, ((category, docId, tf), (idf, word_index))
	words = word_tf.join(word_idf)
	# output format: (word, (category, docId, word_index ,tfidf))
	words = words.map(lambda x: (x[0], (x[1][0][0], x[1][0][1], x[1][1][1], x[1][0][2]*x[1][1][0])))

	#print('\n\n\n ', words.take(5), '\n\n\n')

	# Bring all words in a list for each comment -> output format: ((category, docId), (word_index, tfidf))
	words = words.map(lambda x: ((x[1][1], x[1][0]), [(x[1][2], x[1][3])]) )
	# output format: ((category, docId), list_of(word_index, tfidf))
	words = words.reduceByKey(lambda x, y: x + y)
	# output format: (category, sorted_list with word_index as key and tfidf metric value as value)
	words = words.map(lambda x : (x[0][0], sorted(x[1], key = lambda y : y[0])))

	# Use sparse vector when key is word_index and value is tfidf metric
	data = words.map(lambda x : (x[0], SparseVector(k, [y[0] for y in x[1]], [y[1] for y in x[1]])))

	# Print first 5 lines of rdd
	#res = data.take(5)
	#for r in res:
	#	print(r,'\n')


	#5
	# Add colun names with schema
	df = data.toDF(['category', 'features'])

	#6
	# Transform string labels in integers
	stringIndexer = StringIndexer(inputCol='category', outputCol='label')
	stringIndexer.setHandleInvalid('skip')
	stringIndexerModel = stringIndexer.fit(df)
	df = stringIndexerModel.transform(df)


	# Do train-test split
	seed = 3400043
	fractions = df.select("category").distinct().withColumn("fraction", lit(0.75)).rdd.collectAsMap()
	train_set = df.sampleBy("category", fractions, seed=seed) #.cache()
  	# Subtracting 'train' from original 'data' to get test set
	test_set = df.subtract(train_set)

	# Get number of documents for each set
	print('\n\nSize of train set: ', train_set.count())
	print('\n\nSize of test set: ', test_set.count())
	# Get number of documents for each category for each set
	print('\n\nNo of rows for each category in train set:\n')
	train_set.groupBy("category").count().show()
	print('\n\nNo of rows for each category in test set:\n')
	test_set.groupBy("category").count().show()

	# Get unique categories
	unique_cat = train_set.select("category").distinct().collect()

	#7
	# input layer:k size, output layer:unique_cat size
	layers = [k, 200, len(unique_cat)]

	# Trainer
	trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=seed)

	#testing
	train_set.show()

	start_time = time.time()
	# Train the model
	model = trainer.fit(train_set)

	print("--- Time to train model: {} seconds ---".format(time.time() - start_time))

	# compute accuracy on the test set
	result = model.transform(test_set)
	predictionAndLabels = result.select("prediction", "label")
	evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
	print("\n\n\nTest set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

	# Stop the session
	spark.stop()

