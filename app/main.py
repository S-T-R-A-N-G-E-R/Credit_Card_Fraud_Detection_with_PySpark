from fastapi import FastAPI, UploadFile, File, Request
from pydantic import create_model
from typing import Optional
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col as spark_col, udf
from pyspark.sql.types import DoubleType
from pyspark.ml import PipelineModel
from pyspark.ml.classification import GBTClassificationModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from contextlib import asynccontextmanager
import os, time, logging, traceback
from pydantic import BaseModel, create_model
from typing import Optional

# ==================================================
# UDF for fraud probability
# ==================================================
def extract_prob(v):
    try:
        return float(v[1])
    except Exception:
        return None

extract_prob_udf = udf(extract_prob, DoubleType())


# ==================================================
# Logging setup
# ==================================================
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "api.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)



# ==================================================
# Prometheus metrics
# ==================================================
REQUEST_COUNT = Counter("api_requests_total", "Total API Requests", ["endpoint"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])
FRAUD_PROBABILITY = Histogram("fraud_probability", "Distribution of fraud probability",
                              buckets=[0,0.1,0.25,0.5,0.75,0.9,1.0])


# ==================================================
# Globals for Spark + Models
# ==================================================
spark = None
pipeline_model = None
gbt_model = None


# ==================================================
# Lifespan handler
# ==================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global spark, pipeline_model, gbt_model
    print("🔄 Initializing Spark + Models...")
    spark = SparkSession.builder.appName("CreditCardFraudAPI").getOrCreate()
    pipeline_model = PipelineModel.load("models/preprocessing_pipeline")
    gbt_model = GBTClassificationModel.load("models/gbt_fraud_model")

    yield

    print("🛑 Stopping Spark...")
    spark.stop()

# ==================================================
# FastAPI app
# ==================================================
app = FastAPI(lifespan=lifespan)

# ==================================================
# Compatibility fix for pandas >= 2.0
# ==================================================
if not hasattr(pd.DataFrame, "iteritems"):
    pd.DataFrame.iteritems = pd.DataFrame.items


# ==================================================
# Load pipeline & model only once
# ==================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_PATH = os.path.join(BASE_DIR, "models", "preprocessing_pipeline")
MODEL_PATH = os.path.join(BASE_DIR, "models", "gbt_fraud_model")

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
# Compatibility fix for pandas >= 2.0
# ==================================================
if not hasattr(pd.DataFrame, "iteritems"):
    pd.DataFrame.iteritems = pd.DataFrame.items


# ==================================================
# Load pipeline & model only once
# ==================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_PATH = os.path.join(BASE_DIR, "models", "preprocessing_pipeline")
MODEL_PATH = os.path.join(BASE_DIR, "models", "gbt_fraud_model")

# ==================================================
# Middleware for latency logging
# ==================================================
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    endpoint = request.url.path
    REQUEST_COUNT.labels(endpoint=endpoint).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(process_time)

    logging.info(f"Endpoint={endpoint}, Latency={process_time:.3f}s")

    return response

# ==================================================
# Prometheus metrics endpoint
# ==================================================
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ==================================================
# Endpoints
# ==================================================
@app.post("/predict_single")
def predict_single(request: TransactionRequest):
    start_time = time.time()
    try:
        data = request.model_dump()   # ✅ Pydantic v2 fix
        df = pd.DataFrame([data])
        sdf = spark.createDataFrame(df)

        prepared = pipeline_model.transform(sdf)
        preds = gbt_model.transform(prepared)

        preds = preds.withColumn("fraud_probability", extract_prob_udf(spark_col("probability")))
        result = preds.select("TransactionID", "fraud_probability", "prediction").toPandas()

        result["fraud"] = result["prediction"].apply(lambda x: bool(int(x)))
        result.drop(columns=["prediction"], inplace=True)

        latency = round(time.time() - start_time, 3)
        logging.info(
            f"Endpoint=/predict_single, Latency={latency}s, "
            f"TransactionID={result['TransactionID'].iloc[0]}, "
            f"FraudProb={result['fraud_probability'].iloc[0]:.4f}, "
            f"Fraud={result['fraud'].iloc[0]}"
        )
        return result.to_dict(orient="records")[0]

    except Exception as e:
        logging.error(f"/predict_single ERROR: {e}")
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        df = pd.read_csv(file.file)

        for col in EXPECTED_FEATURES:
            if col not in df.columns:
                df[col] = 0 if not col.endswith(("CD", "Type", "Info", "domain")) else "missing"

        df = df[EXPECTED_FEATURES]
        sdf = spark.createDataFrame(df)

        prepared = pipeline_model.transform(sdf)
        preds = gbt_model.transform(prepared)

        pdf = preds.withColumn("fraud_probability", extract_prob_udf(spark_col("probability"))) \
                   .withColumn("fraud", spark_col("prediction") == 1) \
                   .select("TransactionID", "fraud_probability", "fraud") \
                   .toPandas()

        latency = round(time.time() - start_time, 3)
        logging.info(
            f"Endpoint=/predict_batch, Latency={latency}s, Rows={len(pdf)}, "
            f"FraudProbRange=({pdf['fraud_probability'].min():.4f}, {pdf['fraud_probability'].max():.4f}), "
            f"SampleIDs={pdf['TransactionID'].head(5).tolist()}"
        )
        return pdf.to_dict(orient="records")

    except Exception as e:
        logging.error(f"/predict_batch ERROR: {e}")
        return {"error": str(e), "traceback": traceback.format_exc()}