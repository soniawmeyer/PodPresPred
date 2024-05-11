
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("PodPres").getOrCreate()
df = spark.read.parquet("data/final_features_human_labels.parquet")

df.show()


#DATA PREP
meta = spark.read.csv("data/podcast_meta.csv", header=True)
meta.show()


# check the podcast_counts frequency distribution 
from pyspark.sql.functions import col, ceil
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession

podcast_counts = df.groupBy("podcast_name_cleaned").count()
total_count = df.count()
podcast_counts = podcast_counts.withColumn("percentage_of_total", col("count") / total_count)

joined_df = podcast_counts.join(meta, podcast_counts.podcast_name_cleaned == meta.podcast, "inner")
# |podcast_name_cleaned|count| percentage_of_total| podcast|reviews|review_ratio|rating|date_collected|
final_df = joined_df.select(
    col("podcast_name_cleaned"),
    col("count"),
    col("percentage_of_total"),
    col("review_ratio")
)
final_df = final_df.withColumn("percent_diff", col("review_ratio") - col("percentage_of_total"))
# if positive need to oversample
# if under then what?

final_df = final_df.filter((col("percent_diff") > 0.01) | (col("percent_diff") < -0.01))

final_df.show()

# podcast_counts_ordered = podcast_counts.orderBy(col("count").desc(), col("podcast_name_cleaned"))
# podcast_counts_ordered.show(podcast_counts_ordered.count(), truncate=False)
# podcast_counts = podcast_counts.withColumn("total_count", ceil(col("count") * 0.8))  # Assuming 80% training rate
# podcast_counts.show(podcast_counts.count(), truncate=False)

spark.exit()



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

# model.save("test_model")