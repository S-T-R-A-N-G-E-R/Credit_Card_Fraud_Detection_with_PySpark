# ------------------------------
# Stage 1: Build environment
# ------------------------------
FROM python:3.10-slim-bullseye

# Install dependencies
RUN apt-get update && apt-get install -y \
    openjdk-11-jre-headless curl wget gnupg git \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-arm64
ENV PATH=$JAVA_HOME/bin:$PATH

RUN apt-get update && apt-get install -y \
    openjdk-11-jre-headless curl wget gnupg git procps \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------
# Install Apache Spark
# ------------------------------
ENV SPARK_VERSION=3.3.2
ENV HADOOP_VERSION=3

RUN curl -sL https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    | tar -xz -C /opt/ \
    && ln -s /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /usr/local/spark


# Add Spark to PATH
ENV PATH=/usr/local/spark/bin:$PATH
ENV SPARK_HOME=/usr/local/spark

# Spark + Hadoop setup
ENV SPARK_HOME=/usr/local/spark
ENV HADOOP_HOME=/usr/local/hadoop
ENV PATH=$SPARK_HOME/bin:$PATH
ENV PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.5-src.zip:$PYTHONPATH
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3


# ------------------------------
# Set workdir & install Python deps
# ------------------------------
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


# Run FastAPI with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
