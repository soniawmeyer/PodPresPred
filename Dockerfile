# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the container
WORKDIR /data/

# Install Java (required for PySpark)
RUN apt-get update && \
    apt-get install -y openjdk-11-jre-headless && \
    apt-get clean;

# Install PySpark
RUN pip install pyspark numpy pandas

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Run PySpark script when the container launches
# CMD ["spark-submit", "feature_extraction.py"]
# CMD ["spark-submit", "data_cleaning_final_step.py"]
# CMD ["spark-submit", "parquet_to_csv.py"]
# CMD ["spark-submit", "add_labels.py"]
# CMD ["spark-submit", "model training.ipynb"]
CMD ["spark-submit", "model_training_lr_cont.py"]