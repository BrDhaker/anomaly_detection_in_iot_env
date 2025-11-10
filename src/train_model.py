import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
data = pd.read_csv("Data/data.csv")

# Select the most relevant features :
features = ['temperature', 'vibration', 'humidity', 'energy_consumption', 'predicted_remaining_life', 'downtime_risk']
X = data[features]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the Isolation Forest model (unsupervised anomaly detection)
# Contamination is the proportion of outliers in the data set
model = IsolationForest(contamination=0.08916, random_state=42 , n_estimators=200)
model.fit(X_scaled)

# Save the model and scaler
# joblib.dump(model, "models/anomaly_model_test.pkl")
# joblib.dump(scaler, "models/scaler_test.pkl")

# print("Model and scaler saved!")



# Evaluate on a subset :
predictions = model.predict(X_scaled)
anomaly_scores = model.decision_function(X_scaled)

# Map predictions: -1 is anomaly, 1 is normal
data['predicted_anomaly'] = predictions
data['anomaly_score'] = anomaly_scores

# Save predictions for analysis
# data.to_csv("Output/predictions_test.csv", index=False)

iso_predictions = model.predict(X_scaled)
# Convert predictions: -1 (anomaly) to 1, 1 (normal) to 0 for consistency with anomaly_flag
iso_predictions_binary = (iso_predictions == -1).astype(int)

# Actual anomaly_flag data to compare with
y_true = data['anomaly_flag'].values

print(f"\nIsolation Forest Performance:")
print(classification_report(y_true, iso_predictions_binary))
# cm = confusion_matrix(y_true, iso_predictions_binary)
# Show confusion matrix Figure
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
# plt.title(f"Isolation ForestConfusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.show()

# Print F1-score
from sklearn.metrics import f1_score

f1 = f1_score(y_true, iso_predictions_binary)
print(f"Isolation Forest F1-Score: {f1:.4f}")