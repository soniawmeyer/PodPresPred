
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("PodPres").getOrCreate()
# df = spark.read.parquet("test_features.parquet")


from pyspark.ml import PipelineModel
loaded_model = PipelineModel.load("test_model")


# #PREDICTIONS
# # Assume df_new is your new data
# predictions = model.transform(df_new)
# predictions.select("probability", "prediction").show()



# #EVALUATE
# from pyspark.ml.evaluation import BinaryClassificationEvaluator

# evaluator = BinaryClassificationEvaluator()
# print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
