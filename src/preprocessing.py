import pandas as pd
import re
import pickle
from pathlib import Path

def preprocess_data(df):
    print("\n🧹 Preprocessing data...")
    
    # Clean data
    df['Popular Tags'] = df['Popular Tags'].apply(lambda x: eval(x) if isinstance(x, str) else [])
    df['Game Features'] = df['Game Features'].apply(lambda x: eval(x) if isinstance(x, str) else [])
    df = df.dropna(subset=['Title', 'Game Description'])
    
    # Feature engineering
    def create_features(row):
        tags = ' '.join([str(tag).lower() for tag in row['Popular Tags']])
        features = ' '.join([str(feat).lower() for feat in row['Game Features']])
        desc = re.sub(r'[^\w\s]'), '', str(row['Game Description']).lower()
        return f"{row['Title'].lower()} {tags} {tags} {features} {desc}"
    
    df['enhanced_features'] = df.apply(create_features, axis=1)
    
    # Save processed data
    processed_path = Path(__file__).parent.parent / 'data' / 'processed_data.pkl'
    with open(processed_path, 'wb') as f:
        pickle.dump(df, f)
    
    print(f"✅ Processed {len(df)} records")
    return df

if __name__ == "__main__":
    from src.ingestion import load_data
    df = load_data()
    preprocess_data(df)
    print("\nData preprocessing complete!")