# ==================================================
# Base image
# ==================================================
FROM python:3.10-slim

# Avoid buffering
ENV PYTHONUNBUFFERED=1

# Install Java (needed for Spark)
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

RUN apt-get update && apt-get install -y \
    openjdk-11-jdk curl wget gnupg git \
    && rm -rf /var/lib/apt/lists/*

# ==================================================
# Install Python dependencies
# ==================================================
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ==================================================
# Copy project code
# ==================================================
COPY .. .

# Spark environment variables
ENV SPARK_HOME=/usr/local/spark
ENV PATH=$SPARK_HOME/bin:$PATH

# ==================================================
# Run API
# ==================================================
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
