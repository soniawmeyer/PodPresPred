
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Election Prediction") \
    .getOrCreate()


from pyspark.sql.functions import input_file_name


# #LOAD DATA
# load all txt files in file path
df = spark.read.text("/Users/Sonia/Library/CloudStorage/GoogleDrive-sonia.meyer@sjsu.edu/Shared\ drives/DATA228/Data/podscribe\ app/ben\ shapiro/*.txt") \
        .withColumn("file_name", input_file_name()) #add file path as col name
# df.show()
# df_size = df.count()
# print("DataFrame size:", df_size)

# #PREPROCESSING
from pyspark.sql.functions import regexp_replace, lower, col

df_clean = df.withColumn("text", regexp_replace("value", "[^a-zA-Z0-9\s]", "")) #remove symbols
df_clean = df_clean.withColumn("text", lower(col("text"))) #lowercase
df_clean = df_clean.filter(col('text').contains('biden') | col('text').contains('trump')) #filter for trump/biden
# df_clean.show()
# df_size = df_clean.count()
# print("DataFrame size:", df_size)

#FEATURE EXTRACTION
from pyspark.ml.feature import Tokenizer, StopWordsRemover

tokenizer = Tokenizer(inputCol="text", outputCol="words") #tokenizer to works
df_words = tokenizer.transform(df_clean)

remover = StopWordsRemover(inputCol="words", outputCol="filtered") #remove stop words
df_features = remover.transform(df_words)
df_features.show()

### missing, find +/-100 tokens around trumps / biden


#SENTIMENT ANALYSIS
# Example using a simple lexicon approach
afinn = spark.read.csv("lexicons/Afinn.csv", header=True, inferSchema=True).rdd.collectAsMap()
broadcasted_lexicon = spark.sparkContext.broadcast(afinn)

from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

def sentiment_score(words):
    score = sum(broadcasted_lexicon.value.get(word, 0) for word in words)
    return score

sentiment_udf = udf(sentiment_score, IntegerType())
df_features = df_features.withColumn("sentiment_score", sentiment_udf(df_features["filtered"]))
df_features.show()


#TRAIN MODEL

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml import Pipeline

# Feature transformation
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures") #takes col of tokenized words
idf = IDF(inputCol="rawFeatures", outputCol="features") #inverse document frequency

# Classifier
lr = LogisticRegression(maxIter=10, regParam=0.001)

# Build the pipeline
pipeline = Pipeline(stages=[hashingTF, idf, lr])

# Train the model
# model = pipeline.fit(df_features) #missing label col



# #PREDICTIONS
# # Assume df_new is your new data
# predictions = model.transform(df_new)
# predictions.select("probability", "prediction").show()



# #EVALUATE
# from pyspark.ml.evaluation import BinaryClassificationEvaluator

# evaluator = BinaryClassificationEvaluator()
# print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
