
#INITIALIZE SPARK SESSION
from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .appName("PodPres") \
    .config("spark.executor.memory", "4g") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.executor.memoryOverhead", "1g") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .getOrCreate()

#LOAD DATA
df_features = spark.read.parquet("cleaneddata.parquet")

from pyspark.sql.functions import monotonically_increasing_id, row_number, col
from pyspark.sql.window import Window
# Add a row index to the DataFrame
df_indexed = df_features.withColumn("row_index", monotonically_increasing_id())
# Define a window spec for ordering by the index
windowSpec = Window.orderBy("row_index")

# Add a consecutive row number (this will ensure consecutive numbering)
df_consecutive = df_indexed.withColumn("row_num", row_number().over(windowSpec))


# #data stats
# df_features.show() #['podcast_name_cleaned', 'cleaned_transcript', 'filtered_tokens']
# column_names = df_features.columns
# print(column_names)
# row_count = df_features.count()
# print(f"The number of rows in the DataFrame is: {row_count}")


from pyspark.sql.functions import explode, col, when, collect_list
from pyspark.sql.window import Window
import pyspark.sql.functions as F


# Explode tokens into separate rows
df_exploded = df_consecutive.select(
    "podcast_name_cleaned", 
    "row_num", 
    explode("filtered_tokens").alias("token")
)

# Filter out null, empty or whitespace-only tokens
df_filtered = df_exploded.filter(df_exploded.token.isNotNull() & (df_exploded.token != "") & (~F.trim(df_exploded.token).rlike("^\\s*$")))

# Define a window spec for assigning indexes within each podcast group
window_spec = Window.partitionBy("podcast_name_cleaned").orderBy("row_num")

# Assign indexes within each group
df_indexed = df_filtered.withColumn("index", F.row_number().over(window_spec))

# Calculate group id for segments of 200 tokens
df_segmented = df_indexed.withColumn("segment_id", ((col("index") - 1) / 200).cast("integer"))

# Group by podcast name and segment_id, and collect tokens into lists
df_final = df_segmented.groupBy("podcast_name_cleaned", "segment_id").agg(collect_list("token").alias("segment"))

# # Show results
# df_final.show(truncate=False) #['podcast_name_cleaned', 'segment_id', 'segment']
# column_names = df_final.columns
# print(column_names)
# row_count = df_final.count()
# print(f"The number of rows in the DataFrame is: {row_count}")


from pyspark.sql.functions import udf, col, array_contains
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as F

# Define a UDF to check for token presence in the list
def contains_token(tokens, token_to_check):
    return 1 if token_to_check in (token for token in tokens) else 0

contains_token_udf = udf(contains_token, IntegerType())

# Assuming df_final is the DataFrame obtained from the previous step with columns 'podcast_name_cleaned', 'segment_id', 'segment'

# Apply the UDF to check for 'trump' and 'biden' mentions
df_with_flags = df_final.withColumn(
    "trump_mention", contains_token_udf("segment", F.lit("trump"))
).withColumn(
    "biden_mention", contains_token_udf("segment", F.lit("biden"))
)
df_with_flags = df_with_flags.filter((col("trump_mention") != 0) | (col("biden_mention") != 0))

###############
#what about first names? sleepy joe?
###############


# Show results to verify
# df_with_flags.show(truncate=False) #['podcast_name_cleaned', 'segment_id', 'segment', 'trump_mention', 'biden_mention']
# column_names = df_with_flags.columns
# print(column_names)
# row_count = df_with_flags.count()
# print(f"The number of rows in the DataFrame is: {row_count}")


df_with_flags.write.mode("overwrite").parquet("final_cleaned.parquet")

spark.stop()