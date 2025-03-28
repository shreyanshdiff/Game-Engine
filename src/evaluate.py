import pandas as pd
from pathlib import Path

def evaluate_model():
    print("\n📊 Evaluating model...")
    data_path = Path(__file__).parent.parent / 'data' / 'processed_data.pkl'
    
    with open(data_path, 'rb') as f:
        df = pickle.load(f)
    
    # Sample evaluation
    test_games = ["Baldur's Gate 3", "Counter-Strike", "The Witcher 3"]
    
    for game in test_games:
        recs = recommend(game, df)
        print(f"\nRecommendations for {game}:")
        print(recs[['Title', 'Similarity']].to_markdown(index=False))
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    from src.recommend import recommend
    import pickle
    evaluate_model()