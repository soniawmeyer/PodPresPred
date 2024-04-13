
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

df.write.parquet("test_cleaning.parquet")