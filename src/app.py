from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the API
app = FastAPI(title="Anomaly Detection API", description="API for detecting anomalies in IoT sensor data")

# Load the trained model and scaler
try:
    model = joblib.load("models/anomaly_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    logger.info("Model and scaler loaded successfully")
except FileNotFoundError:
    logger.error("Model or scaler file not found. Please run src/train_model.py first.")
    model = None
    scaler = None

# Define the input schema
class SensorData(BaseModel):
    temperature: float
    vibration: float
    humidity: float
    energy_consumption: float
    predicted_remaining_life: float
    downtime_risk: float

@app.post("/predict")
def predict(data: SensorData):
    if model is None or scaler is None:
        return {"error": "Model not loaded"}

    # Log the incoming data
    logger.info(f"Received data: {data}")

    # Convert input to numpy array
    input_data = np.array([[data.temperature, data.vibration, data.humidity, data.energy_consumption, data.predicted_remaining_life, data.downtime_risk]])

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Predict anomaly score
    anomaly_score = model.decision_function(input_scaled)[0]
    is_anomaly = model.predict(input_scaled)[0]  # 1 = normal, -1 = anomaly

    # Log the prediction
    logger.info(f"Anomaly score: {anomaly_score}, Is anomaly: {is_anomaly == -1}")

    return {
        "anomaly_score": anomaly_score,
        "is_anomaly": bool(is_anomaly == -1)  # True if anomaly
    }

@app.get("/")
def read_root():
    return {"message": "Anomaly Detection API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)