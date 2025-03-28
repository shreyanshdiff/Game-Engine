import pickle
from pathlib import Path

def recommend(title, df=None, n_recommendations=5):
    """Main recommendation function"""
    if df is None:
        data_path = Path(__file__).parent.parent / 'data' / 'processed_data.pkl'
        with open(data_path, 'rb') as f:
            df = pickle.load(f)
    
    model_path = Path(__file__).parent.parent / 'models'
    
    try:
        with open(model_path / 'pipeline.pkl', 'rb') as f:
            pipeline = pickle.load(f)
        with open(model_path / 'knn_model.pkl', 'rb') as f:
            knn = pickle.load(f)
        
        matches = df[df['Title'].str.contains(title, case=False)]
        if len(matches) == 0:
            return df[['Title', 'Link']].head(n_recommendations)
        
        idx = matches.index[0]
        query = pipeline.transform([df.loc[idx, 'enhanced_features']])
        
        distances, indices = knn.kneighbors(query, n_neighbors=n_recommendations+1)
        
        results = []
        for i, distance in zip(indices[0][1:], distances[0][1:]):
            game = df.iloc[i]
            results.append({
                'Title': game['Title'],
                'Link': game['Link'],
                'Similarity': f"{1 - distance:.3f}",
                'Price': game.get('Discounted Price', game.get('Original Price', 'N/A'))
            })
        
        return pd.DataFrame(results)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return pd.DataFrame()

if __name__ == "__main__":
    import pandas as pd
    from src.preprocessing import preprocess_data
    
    df = preprocess_data()
    test_game = input("Enter game title: ")
    recommendations = recommend(test_game, df)
    
    print("\n🎮 Recommended Games:")
    print(recommendations[['Title', 'Similarity', 'Price']].to_markdown(index=False))