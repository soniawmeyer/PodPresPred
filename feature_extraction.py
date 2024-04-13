

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("PodPres").getOrCreate()
df_features = spark.read.parquet("test_cleaning.parquet")


# # TOPIC MODELING
# from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer

# vectorizer = CountVectorizer(inputCol="filtered", outputCol="features")
# model = vectorizer.fit(filtered_data)
# vectorized_data = model.transform(filtered_data)
# from pyspark.ml.clustering import LDA

# lda = LDA(k=5, maxIter=10)  # k is the number of topics
# model = lda.fit(vectorized_data)

# # Show results
# topics = model.describeTopics(maxTermsPerTopic=10)
# topics.show(truncate=False)

# transformed = model.transform(vectorized_data)
# transformed.show(truncate=False)
# vocab = model.vocabulary
# for index, topic in enumerate(topics.collect()):
#     print(f"Topic {index}:")
#     for word, weight in zip(topic.termIndices, topic.termWeights):
#         print(f"{vocab[word]} {weight}")



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



# from pyspark.ml.feature import VectorAssembler

# # Assuming you have already extracted word tokens, sentiment scores, and topic features
# # Let's say you have columns named "word_tokens", "sentiment_score", and "topic_features"

# # Define the list of input columns to be assembled
# input_columns = ["word_tokens", "sentiment_score", "topic_features"]

# # Create the VectorAssembler instance
# vector_assembler = VectorAssembler(inputCols=input_columns, outputCol="features")

# # Apply the VectorAssembler to your DataFrame
# assembled_df = vector_assembler.transform(your_original_df)

# # Show the DataFrame with the assembled features
# assembled_df.select("features", "label").show(truncate=False)


df.write.parquet("test_features.parquet")