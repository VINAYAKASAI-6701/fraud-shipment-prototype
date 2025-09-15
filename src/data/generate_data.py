import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_shipments(n=500):
    np.random.seed(42)
    
    shipment_ids = [f"SHP{1000+i}" for i in range(n)]
    origins = np.random.choice(['NY', 'LA', 'TX', 'FL', 'IL'], n)
    destinations = np.random.choice(['NY', 'LA', 'TX', 'FL', 'IL'], n)
    weights = np.random.normal(10, 5, n).round(2)
    transit_days = np.random.normal(5, 2, n).astype(int)
    shipment_dates = [datetime.now() - timedelta(days=int(np.random.rand()*30)) for _ in range(n)]
    
    # Introduce some anomalies
    anomaly_flags = np.random.choice([0,1], n, p=[0.95, 0.05])
    weights[anomaly_flags==1] *= np.random.randint(3, 10, anomaly_flags.sum())
    transit_days[anomaly_flags==1] += np.random.randint(5,15, anomaly_flags.sum())
    
    df = pd.DataFrame({
        'shipment_id': shipment_ids,
        'origin': origins,
        'destination': destinations,
        'weight_kg': weights,
        'transit_days': transit_days,
        'shipment_date': shipment_dates,
        'is_fraud': anomaly_flags
    })
    
    return df

if __name__ == "__main__":
    df = generate_shipments(500)
    df.to_csv('data/shipments.csv', index=False)
    print("Generated shipments.csv with 500 rows")
