from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import pickle
from pathlib import Path

def train_model(df):
    print("\n🤖 Training recommendation model...")
    
    # Create pipeline
    pipeline = make_pipeline(
        TfidfVectorizer(max_features=10000, ngram_range=(1, 2)),
        Normalizer()
    )
    
    # Fit model
    X = pipeline.fit_transform(df['enhanced_features'])
    knn = NearestNeighbors(n_neighbors=6, metric='cosine', n_jobs=-1)
    knn.fit(X)
    
    # Save artifacts
    model_path = Path(__file__).parent.parent / 'models'
    model_path.mkdir(exist_ok=True)
    
    with open(model_path / 'pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    with open(model_path / 'knn_model.pkl', 'wb') as f:
        pickle.dump(knn, f)
    
    print("✅ Model training complete!")
    return pipeline, knn

if __name__ == "__main__":
    from src.preprocessing import preprocess_data
    df = preprocess_data()
    train_model(df)
    print("\nModel training complete!")