#INITIALIZE SPARK SESSION
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("PodPres").getOrCreate()

#LOAD DATA
from pyspark.sql.functions import monotonically_increasing_id
df_features = spark.read.parquet("/data/data/final_features.parquet") #['podcast_name_cleaned', 'segment_id', 'segment', 'trump_mention', 'biden_mention']
df_features = df_features.withColumn("segment_id", monotonically_increasing_id())
df_features = df_features.select('segment_id', 'podcast_name_cleaned','features')
# df_features.show()

#ADD LABELS
labels_csv = spark.read.csv("/data/data/easy_labels.csv", header=True)
labels_csv = labels_csv.withColumnRenamed("LABEL", "label")
# labels_csv.show()

assembled_df = df_features.join(labels_csv.select("segment_id", "label"), on="segment_id")
assembled_df.show()

assembled_df.write.mode("overwrite").parquet("final_features_human_labels.parquet")
spark.stop()