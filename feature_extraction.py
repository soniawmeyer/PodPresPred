

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("PodPres").getOrCreate()
df_features = spark.read.parquet("test_cleaning.parquet")


#RANDOM LABELS
from pyspark.sql.functions import rand
df_features = df_features.withColumn("label", (rand() * 2).cast("int"))
df_features.show()



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


# # # TOPIC MODELING
# # from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer

# # vectorizer = CountVectorizer(inputCol="filtered", outputCol="features")
# # model = vectorizer.fit(filtered_data)
# # vectorized_data = model.transform(filtered_data)
# # from pyspark.ml.clustering import LDA

# # lda = LDA(k=5, maxIter=10)  # k is the number of topics
# # model = lda.fit(vectorized_data)

# # # Show results
# # topics = model.describeTopics(maxTermsPerTopic=10)
# # topics.show(truncate=False)

# # transformed = model.transform(vectorized_data)
# # transformed.show(truncate=False)
# # vocab = model.vocabulary
# # for index, topic in enumerate(topics.collect()):
# #     print(f"Topic {index}:")
# #     for word, weight in zip(topic.termIndices, topic.termWeights):
# #         print(f"{vocab[word]} {weight}")



# #WORD2VEC
# from pyspark.ml.feature import Word2Vec
# from pyspark.sql import SparkSession
# from pyspark.sql.types import ArrayType, FloatType

# # Train Word2Vec model
# word2Vec = Word2Vec(vectorSize=100, minCount=0, inputCol="filtered", outputCol="embeddings")
# model = word2Vec.fit(df_features)
# # 847-53

# # Embed tokens
# df_embedded = model.transform(df_features)

# # Show the result
# df_embedded.select("filtered", "embeddings").show(truncate=False)


# # Define a UDF to extract the embedding vectors
# extract_vectors_udf = udf(lambda embeddings: embeddings.toArray().tolist(), ArrayType(FloatType()))

# # Apply the UDF to extract the embedding vectors
# df_with_embeddings = df_embedded.withColumn("embeddings_list", extract_vectors_udf("embeddings"))

# # Add the embedded vectors as a new column to the original DataFrame
# df_features_with_embeddings = df_features.join(df_with_embeddings.select("filtered", "embeddings_list"), on="filtered")

# # Show the result
# df_features_with_embeddings.show(truncate=False)




# from pyspark.ml.feature import VectorAssembler

# Assuming you have already extracted word tokens, sentiment scores, and topic features
# Let's say you have columns named "word_tokens", "sentiment_score", and "topic_features"

# Define the list of input columns to be assembled
input_columns = ["filtered", "sentiment_score", ] #"topic_features"

# Create the VectorAssembler instance
vector_assembler = VectorAssembler(inputCols=input_columns, outputCol="features")

# Apply the VectorAssembler to your DataFrame
assembled_df = vector_assembler.transform(df_features)

# Show the DataFrame with the assembled features
assembled_df.select("features", "label").show(truncate=False)


df_features.write.mode("overwrite").parquet("test_features.parquet")