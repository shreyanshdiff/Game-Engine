from src.ingestion import load_data
from src.preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    print(" Starting Game Recommender Pipeline")
    
    # Run pipeline steps
    df = load_data()
    df = preprocess_data(df)
    train_model(df)
    evaluate_model()
    
    print("\n src execution complete!")

if __name__ == "__main__":
    main()