from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import PipelineModel
from pyspark.ml.classification import GBTClassificationModel
from pyspark.ml.functions import vector_to_array
import os
import traceback


# ==================================================
# Compatibility fix for pandas >= 2.0
# ==================================================
if not hasattr(pd.DataFrame, "iteritems"):
    pd.DataFrame.iteritems = pd.DataFrame.items

# ==================================================
# Paths
# ==================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_PATH = os.path.join(BASE_DIR, "models", "preprocessing_pipeline")
MODEL_PATH = os.path.join(BASE_DIR, "models", "gbt_fraud_model")

# ==================================================
# Load preprocessing pipeline + model once at startup
# ==================================================
print("🔄 Loading preprocessing pipeline...")
pipeline_model = PipelineModel.load(PIPELINE_PATH)

print("🔄 Loading GBT model...")
gbt_model = GBTClassificationModel.load(MODEL_PATH)

# ==================================================
# Spark session
# ==================================================
spark = SparkSession.builder \
    .appName("CreditCardFraudAPI") \
    .getOrCreate()

from pydantic import BaseModel, create_model
from typing import Optional

# ==================================================
# Define request schema dynamically
# ==================================================
# Core base fields
core_fields = {
    "TransactionID": (Optional[int], 0),
    "TransactionDT": (Optional[int], 0),
    "TransactionAmt": (Optional[float], 0.0),
    "ProductCD": (Optional[str], "unknown"),
    "card1": (Optional[int], 0),
    "card2": (Optional[float], 0.0),
    "card3": (Optional[float], 0.0),
    "card4": (Optional[str], "unknown"),
    "card5": (Optional[float], 0.0),
    "card6": (Optional[str], "unknown"),
    "addr1": (Optional[float], 0.0),
    "addr2": (Optional[float], 0.0),
    "dist1": (Optional[float], 0.0),
    "dist2": (Optional[float], 0.0),
    "P_emaildomain": (Optional[str], "unknown"),
    "R_emaildomain": (Optional[str], "unknown"),
    "DeviceType": (Optional[str], "unknown"),
    "DeviceInfo": (Optional[str], "unknown"),
}

# Expand features programmatically
extra_fields = {}

# C1–C14
for i in range(1, 15):
    extra_fields[f"C{i}"] = (Optional[float], 0.0)

# D1–D15
for i in range(1, 16):
    extra_fields[f"D{i}"] = (Optional[float], 0.0)

# M1–M9
for i in range(1, 10):
    extra_fields[f"M{i}"] = (Optional[str], "F")

# V1–V339
for i in range(1, 340):
    extra_fields[f"V{i}"] = (Optional[float], 0.0)

# id_01–id_38
for i in range(1, 39):
    extra_fields[f"id_{i:02d}"] = (Optional[float], 0.0)

# Merge all fields
all_fields = {**core_fields, **extra_fields}

# ✅ Correct way in Pydantic v2
TransactionRequest = create_model(
    "TransactionRequest",
    **all_fields
)


# Full list of expected features (must match training pipeline schema)
EXPECTED_FEATURES = [
    # Transaction info
    "TransactionID", "TransactionDT", "TransactionAmt", "ProductCD",

    # Card info
    "card1", "card2", "card3", "card4", "card5", "card6",

    # Address + distance
    "addr1", "addr2", "dist1", "dist2",

    # Emails
    "P_emaildomain", "R_emaildomain",

    # Count features
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13", "C14",

    # Time deltas
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9",
    "D10", "D11", "D12", "D13", "D14", "D15",

    # Match flags
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",

    # V1 → V339
] + [f"V{i}" for i in range(1, 340)] + [

    # Identity features
    "id_01", "id_02", "id_03", "id_04", "id_05", "id_06", "id_07", "id_08", "id_09",
    "id_10", "id_11", "id_12", "id_13", "id_14", "id_15", "id_16", "id_17", "id_18", "id_19",
    "id_20", "id_21", "id_22", "id_23", "id_24", "id_25", "id_26", "id_27", "id_28", "id_29",
    "id_30", "id_31", "id_32", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",

    # Device info
    "DeviceType", "DeviceInfo"
]


# ==================================================
# FastAPI app
# ==================================================
app = FastAPI()

@app.post("/predict_single")
def predict_single(request: TransactionRequest):
    try:
        # Convert request → dict
        data = request.dict()

        # Pandas DF → Spark DF
        df = pd.DataFrame([data])
        sdf = spark.createDataFrame(df)

        # Transform with pipeline + model
        prepared = pipeline_model.transform(sdf)
        preds = gbt_model.transform(prepared)

        # Convert probability vector to array & extract fraud prob
        preds = preds.withColumn("probability_array", vector_to_array(col("probability")))
        preds = preds.withColumn("fraud_probability", preds["probability_array"].getItem(1))

        # Final result
        result = preds.select("TransactionID", "fraud_probability", "prediction").toPandas()

        # Add fraud flag
        result["fraud"] = result["prediction"].apply(lambda x: bool(int(x)))

        # Drop raw prediction (optional — cleaner output)
        result.drop(columns=["prediction"], inplace=True)

        return result.to_dict(orient="records")[0]

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    try:
        # Read CSV into Pandas
        df = pd.read_csv(file.file)

        # Fill missing features with defaults
        for f in EXPECTED_FEATURES:
            if f not in df.columns:
                df[f] = 0 if not f.endswith(("CD", "Type", "Info", "domain")) else "missing"

        # Reorder columns
        df = df[EXPECTED_FEATURES]

        # Convert to Spark DF
        sdf = spark.createDataFrame(df)

        # Apply preprocessing + predict
        prepared = pipeline_model.transform(sdf)
        preds = gbt_model.transform(prepared)

        # Extract fraud probability correctly
        preds = preds.withColumn(
            "probability_array", vector_to_array(col("probability"))
        ).withColumn(
            "fraud_probability", col("probability_array").getItem(1)
        )

        # Collect results
        pdf = preds.select("TransactionID", "fraud_probability", "prediction").toPandas()
        pdf["fraud"] = pdf["prediction"].apply(lambda x: bool(int(x)))
        pdf.drop(columns=["prediction"], inplace=True)

        return pdf.to_dict(orient="records")

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}
