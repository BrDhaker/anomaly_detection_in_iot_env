## Project Structure

```
From Model to Production/
├── Data/
│   └── data.csv
├── models/
│   ├── anomaly_model.pkl
│   └── scaler.pkl
├── Output/
│   ├── predictions.csv
│   └── feature_distributions.png
├── src/
│   ├── train_model.py
│   ├── app.py
│   └── stream_data.py
├── analysis.ipynb
├── requirements.txt
└── README.md
```

## Installation

1. Clone or download this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
The usage is too simple, in three steps:

### 1. Train the Model

Run the training script to build and save the anomaly detection model:

```bash
python src/train_model.py
```

This will:
- Load the dataset (`Data/data.csv`)
- Train an Isolation Forest model on temperature, vibration, and humidity features
- Save the model (`models/anomaly_model.pkl`) and scaler (`models/scaler.pkl`)
- Generate predictions and save to `Output/predictions.csv`

### 2. Start the API

Run the FastAPI server:

```bash
python src/app.py
```

Or using uvicorn:

```bash
uvicorn src.app:app --reload
```

The API will be available at `http://127.0.0.1:8000`

### 3. Test the API

You can test the API manually:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{'temperature': 85.0, 'vibration': 35.0, 'humidity': 75.0, 'energy_consumption': 450.0, 'predicted_remaining_life': 85.0, 'downtime_risk': 0.2}'
```

#
Expected response:
```json
{
  "anomaly_score": 0.01804829394141738,
  "is_anomaly": false
}
```
To stream data continuously, run:

```bash
python src/stream_data.py
```
>if the cloud endpoint is offline, comment the cloud data loading section and uncomment the local data loading section in `stream_data.py`& then run the script again.

### Test if the azure endpoint is online for a faster response example : 
```bash
curl -X POST "https://anomalydetectionwebapp2025.azurewebsites.net/predict" -H "Content-Type: application/json" -d '{"temperature": 85.0, "vibration": 35.0, "humidity": 75.0, "energy_consumption": 450.0, "predicted_remaining_life": 85.0, "downtime_risk": 0.2}'

```
### 5. Run Analysis Notebook

Open and run the comprehensive analysis notebook:

```bash
jupyter notebook anomaly_detection.ipynb
```

This notebook includes:
- Data exploration and visualization
- Model training and evaluation
- **Model comparison** between Isolation Forest, One-Class SVM, and Local Outlier Factor
- Performance analysis and insights
- Anomaly detection results

## Architecture

```
Sensor Data Stream → REST API → Preprocessing → Anomaly Model → Prediction Response
```

- **Data Ingestion**: Simulated via streaming script
- **Processing**: Normalization using StandardScaler
- **Model**: Isolation Forest for anomaly detection ( The best model after comparison with One-Class SVM and Local Outlier Factor)
- **API**: FastAPI for RESTful interface

## Dataset

The project uses the "Smart Manufacturing IoT-Cloud Monitoring Dataset" from Kaggle:
- Features: 13
- Target: anomaly_flag (for evaluation)
- Size: 100,000 records

Download from: https://www.kaggle.com/datasets/ziya07/smart-manufacturing-iot-cloud-monitoring-dataset

## Evaluation

The model is evaluated using:
- Precision, Recall, F1-score (compared to anomaly_flag)
- Anomaly scores from Isolation Forest, One-Class SVM, and Local Outlier Factor


## License

This project is for educational purposes.