
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("PodPres").getOrCreate()
df = spark.read.parquet("test_features.parquet")



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
