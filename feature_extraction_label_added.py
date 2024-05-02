#INITIALIZE SPARK SESSION
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("PodPres").getOrCreate()

#LOAD DATA
df_features = spark.read.parquet("final_cleaned.parquet") #['podcast_name_cleaned', 'segment_id', 'segment', 'trump_mention', 'biden_mention']
# df_features = df_features.select('file_name', 'segment')
# df_features = df_features.limit(3) #remove later

# Show the updated DataFrame
# df_features.show()
# column_names = df_features.columns
# print(column_names)
# row_count = df_features.count()
# print(f"The number of rows in the final DataFrame is: {row_count}")

#RANDOM LABELS
from pyspark.sql.functions import rand
df_features = df_features.withColumn("label", (rand() * 2).cast("int"))

#SENTIMENT ANALYSIS
afinn = spark.read.csv("lexicons/Afinn.csv", header=True, inferSchema=True).rdd.collectAsMap()
broadcasted_lexicon = spark.sparkContext.broadcast(afinn)

from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

def sentiment_score(words):
    score = sum(broadcasted_lexicon.value.get(word, 0) for word in words)
    return score

sentiment_udf = udf(sentiment_score, IntegerType())
df_features = df_features.withColumn("sentiment_score", sentiment_udf(df_features["segment"]))

# Perform descriptive statistics on the 'sentiment_score' column
df_features.describe("sentiment_score").show()

import matplotlib.pyplot as plt

pdf = df_features.toPandas()
# Plot the histogram
plt.hist(pdf['sentiment_score'], bins=50, alpha=0.75)
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title('Histogram of Sentiment Scores')
plt.grid(True)
plt.show()

from pyspark.sql.functions import when
#  sentiment_score > 20 as positive，< 0 as negtive，= 0 as neutral
df_features = df_features.withColumn(
    "label0",
    when(df_features.sentiment_score > 20, "Positive")
    .when(df_features.sentiment_score < 5, "Negative")
    .otherwise("Neutral")
)
df_features.show()

df_features.groupBy("label0").count().show()

#find all segments where the sentiment_score is equal to any score
#target_score = 5
#filtered_df = df_features.filter(df_features.sentiment_score == target_score)
# Display the 'segment' column of the filtered DataFrame
#filtered_df.select("segment").show(truncate=False)


# First, remove the 'label' column
df_features = df_features.drop("label")
# Then, rename 'label0' to 'label'
df_features = df_features.withColumnRenamed("label0", "label")
# Display the updated DataFrame to verify that the changes have been applied correctly
df_features.show()


# TOPIC MODELING
from pyspark.ml.feature import CountVectorizer
cv = CountVectorizer(inputCol="segment", outputCol="features", vocabSize=100, minDF=3.0)
cv_model = cv.fit(df_features)
vectorized_data = cv_model.transform(df_features)

# Train LDA model
from pyspark.ml.clustering import LDA
lda = LDA(k=5, maxIter=10)
model = lda.fit(vectorized_data)

# Show topics
topics = model.describeTopics(maxTermsPerTopic=5)
topics.show(truncate=False)

# Transform original data
transformed = model.transform(vectorized_data)
transformed.show(truncate=False)

# Use the vocabulary from CountVectorizerModel
vocab = cv_model.vocabulary

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

# Define a UDF to convert feature indices back to words
def indices_to_words(vec):
    return [vocab[i] for i in vec.indices]

indices_to_words_udf = udf(indices_to_words, ArrayType(StringType()))

# Apply UDF to the feature column to create a new 'used_words' column
df_features_with_used_words = vectorized_data.withColumn("used_words", indices_to_words_udf("features"))



#WORD2VEC ON segment TOKENS
from pyspark.ml.feature import Word2Vec
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, FloatType

# Train Word2Vec model
word2Vec = Word2Vec(vectorSize=100, minCount=0, inputCol="segment", outputCol="embeddings")
model = word2Vec.fit(df_features)

# Embed tokens
df_embedded = model.transform(df_features)

# Show the result
df_embedded.select("segment", "embeddings").show(truncate=False)

df_embedded.show()

# # Define a UDF to extract the embedding vectors
extract_vectors_udf = udf(lambda embeddings: embeddings.toArray().tolist(), ArrayType(FloatType()))

# Apply the UDF to extract the embedding vectors
df_with_embeddings = df_embedded.withColumn("embeddings_list", extract_vectors_udf("embeddings"))

# Add the embedded vectors as a new column to the original DataFrame
df_features_with_embeddings = df_features_with_used_words.join(df_with_embeddings.select("segment", "embeddings_list"), on="segment")

#CONVERT EMBEDDING LIST TO DENSE VECTOR
from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors, VectorUDT

# Define a UDF to convert array<float> to DenseVector
array_to_vector_udf = udf(lambda arr: Vectors.dense(arr), VectorUDT())

# Apply the UDF to create the DenseVector column
df_features_with_embeddings = df_features_with_embeddings.withColumn("embeddings_vector", array_to_vector_udf(df_features_with_embeddings["embeddings_list"]))







#W2V ON TOPIC MODEL
word2Vec = Word2Vec(vectorSize=100, minCount=0, inputCol="used_words", outputCol="embeddings_tm")
model = word2Vec.fit(df_features_with_embeddings)

# Embed tokens
df_embedded = model.transform(df_features_with_embeddings)

# Show the result
df_embedded.select("used_words", "embeddings_tm").show(truncate=False)

df_embedded.show()

# # Define a UDF to extract the embedding vectors
extract_vectors_udf = udf(lambda embeddings: embeddings.toArray().tolist(), ArrayType(FloatType()))

# Apply the UDF to extract the embedding vectors
df_with_embeddings = df_embedded.withColumn("embeddings_list_tm", extract_vectors_udf("embeddings_tm"))

# Add the embedded vectors as a new column to the original DataFrame
df_features_with_embeddings = df_features_with_embeddings.join(df_with_embeddings.select("used_words", "embeddings_list_tm"), on="used_words")

#CONVERT EMBEDDING LIST TO DENSE VECTOR
from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors, VectorUDT

# Define a UDF to convert array<float> to DenseVector
array_to_vector_udf = udf(lambda arr: Vectors.dense(arr), VectorUDT())

# Apply the UDF to create the DenseVector column
df_features_with_embeddings = df_features_with_embeddings.withColumn("embeddings_vector_tm", array_to_vector_udf(df_features_with_embeddings["embeddings_list_tm"]))

# # Show the result
df_features_with_embeddings.show()
column_names = df_features_with_embeddings.columns
print(column_names)





#VECTOR ASSEMBLY
from pyspark.ml.feature import VectorAssembler

# ['segment', 'file_name', 'label', 'sentiment_score', 'features', 'used_words', 'embeddings_list']
input_columns = ["sentiment_score", "embeddings_vector",'embeddings_vector_tm']

# Create the VectorAssembler instance
vector_assembler = VectorAssembler(inputCols=input_columns, outputCol="final_features")

# Apply the VectorAssembler to your DataFrame
assembled_df = vector_assembler.transform(df_features_with_embeddings)


# assembled_df = assembled_df.select('features', 'used_words', 'embeddings_list_tm','embeddings_vector_tm')
# assembled_df = assembled_df.select('embeddings_list', 'embeddings_vector')


assembled_df = assembled_df.select('podcast_name_cleaned', 'trump_mention','biden_mention','final_features', 'label')
assembled_df = assembled_df.withColumnRenamed("final_features", "features")
assembled_df.show()

assembled_df.write.mode("overwrite").parquet("final_features.parquet")
spark.stop()
