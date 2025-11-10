import time
import requests
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the dataset
# file_path = "../Data/data.csv"
# try:
#     data = pd.read_csv(file_path)
#     logger.info(f"Dataset loaded with {len(data)} records")
# except FileNotFoundError:
#     logger.error(f"Dataset file {file_path} not found")
#     exit(1)





# Load the dataset from Azure Blob Storage
from azure.storage.blob import BlobServiceClient
import io
# Azure Blob Storage settings
AZURE_CONNECTION_STRING = "" 
CONTAINER_NAME = "datasets"
BLOB_NAME = "data.csv"

try:
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    blob_client = blob_service_client.get_container_client(CONTAINER_NAME).get_blob_client(BLOB_NAME)
    blob_data = blob_client.download_blob().readall()
    data = pd.read_csv(io.BytesIO(blob_data))
    logger.info(f"Dataset loaded from Azure Blob with {len(data)} records")
except Exception as e:
    logger.error(f"Failed to load dataset from Azure Blob: {e}")
    exit(1)


# Select relevant columns for streaming
stream_data = data[['temperature', 'vibration', 'humidity', 'energy_consumption', 'predicted_remaining_life', 'downtime_risk']]

# API endpoint
# url = "http://127.0.0.1:8000/predict"
# To use the deployed Azure endpoint instead:
url = "https://anomalydetectionwebapp2025.azurewebsites.net/predict"

# Number of records to stream (to avoid infinite loop, we will limit to first 1000 for just a demo)
max_records = min(1000, len(stream_data))

logger.info(f"Starting to stream {max_records} records...")

# Simulate streaming
for index in range(max_records):
    row = stream_data.iloc[index]

    # Convert row to JSON
    payload = {
        "temperature": float(row['temperature']),
        "vibration": float(row['vibration']),
        "humidity": float(row['humidity']),
        "energy_consumption": float(row['energy_consumption']),
        "predicted_remaining_life": float(row['predicted_remaining_life']),
        "downtime_risk": float(row['downtime_risk'])
    }

    try:
        # Send data to the API
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Record {index+1}: Sent {payload}, Received: {result}")
        else:
            logger.error(f"Record {index+1}: API error {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Record {index+1}: Request failed: {e}")

    # Simulate a delay (e.g., 1 second between data points)
    time.sleep(1)

logger.info("Streaming completed.")