import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

def train_model(csv_path='data/shipments.csv', save_path='models/anomaly_model.pkl'):
    # Load shipment data
    df = pd.read_csv(csv_path)
    
    # Features for anomaly detection
    features = ['weight_kg', 'transit_days']
    
    # Train IsolationForest
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(df[features])
    
    # Add predictions to dataframe
    df['anomaly_score'] = model.decision_function(df[features])
    df['predicted_anomaly'] = model.predict(df[features])
    df['predicted_anomaly'] = df['predicted_anomaly'].map({1:0, -1:1})
    
    # Make sure directories exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Save the trained model
    joblib.dump(model, save_path)
    
    # Save predictions CSV
    df.to_csv('data/shipments_with_predictions.csv', index=False)
    print("Model saved and predictions CSV created at data/shipments_with_predictions.csv")

if __name__ == "__main__":
    train_model()
