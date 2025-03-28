import pandas as pd
import os
from pathlib import Path

def load_data():
    print("🚀 Starting data ingestion...")
    data_path = Path(__file__).parent.parent / 'data' / 'merged_data.csv'
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} records")
    return df

if __name__ == "__main__":
    df = load_data()
    print(df[['Title']].head(3))
    print("\nData ingestion complete!")