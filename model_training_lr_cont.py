
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, ceil
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType


spark = SparkSession.builder.appName("PodPres").getOrCreate()
df = spark.read.parquet("data/final_features_human_labels.parquet")
# ['segment_id', 'podcast_name_cleaned', 'features', 'label']


# Assuming 'psypark_df' is your DataFrame and 'column_label' is the name of the column you want to convert
df = df.withColumn("label", col("label").cast(FloatType()))


filteredData = df.select(
    col("features"),
    col("label"))


train_data, test_data = filteredData.randomSplit([0.7, 0.3])

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol='features', labelCol='label')
lrModel = lr.fit(train_data)


lrModel.save("lr_cont_model_sm")

from pyspark.ml.evaluation import RegressionEvaluator

# lrModel = LinearRegressionModel.load("lr_cont_model_sm")

predictions = lrModel.transform(test_data)


evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

