import os
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import GBTClassificationModel
from pyspark.sql.functions import regexp_replace, col, udf
from pyspark.sql.types import DoubleType
import pandas as pd

# Paths
# Paths
# Force BASE_DIR to the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

PIPELINE_PATH = os.path.join(BASE_DIR, "models/preprocessing_pipeline")
MODEL_PATH = os.path.join(BASE_DIR, "models/gbt_fraud_model")
DATA_PATH = os.path.join(BASE_DIR, "data/processed/test.parquet")
OUTPUT_PARQUET_PATH = os.path.join(BASE_DIR, "data/processed/test_predictions.parquet")
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "data/processed/test_predictions.csv")

print("DEBUG paths:")
print("BASE_DIR:", BASE_DIR)
print("PIPELINE_PATH:", PIPELINE_PATH)
print("MODEL_PATH:", MODEL_PATH)
print("DATA_PATH:", DATA_PATH)



# --- Step 1: Start Spark session ---
spark = SparkSession.builder \
    .appName("FraudDetectionInference") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.maxResultSize", "4g") \
    .getOrCreate()

print("🔄 Loading preprocessing pipeline...")
pipeline_model = PipelineModel.load(PIPELINE_PATH)

print("🔄 Loading GBT model...")
gbt_model = GBTClassificationModel.load(MODEL_PATH)

print(f"🔄 Loading data from: {DATA_PATH}")
df = spark.read.parquet(DATA_PATH)
print(f"✅ Data loaded with {df.count()} rows and {len(df.columns)} columns")

# --- Step 2: Normalize column names ---
df = df.toDF(*[c.replace("-", "_") for c in df.columns])
print("✅ Column names normalized (hyphens → underscores)")

# --- Step 3: Apply preprocessing ---
print("🔄 Applying preprocessing pipeline...")
prepared_df = pipeline_model.transform(df)

# --- Step 4: Run inference ---
print("🔄 Running inference with GBT model...")
predictions = gbt_model.transform(prepared_df)

# --- Step 5: Extract fraud probability ---
@udf(DoubleType())
def extract_prob(probability):
    return float(probability[1]) if hasattr(probability, "__getitem__") else None

predictions = predictions.withColumn("fraud_probability", extract_prob(col("probability")))

# Keep only required columns
predictions = predictions.select("TransactionID", "fraud_probability", "prediction")

# --- Step 6: Save outputs ---
print(f"💾 Saving predictions to Parquet: {OUTPUT_PARQUET_PATH}")
predictions.write.mode("overwrite").parquet(OUTPUT_PARQUET_PATH)

print(f"💾 Saving predictions to CSV: {OUTPUT_CSV_PATH}")
predictions.toPandas().to_csv(OUTPUT_CSV_PATH, index=False)

print("✅ Inference complete! Predictions saved to both Parquet and CSV")

spark.stop()
