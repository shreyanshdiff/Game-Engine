import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import re

# Load your data
df = pd.read_csv('merged_data.csv')

# 1. Clean the data properly
print("Initial data shape:", df.shape)

# Handle missing values correctly
df['Popular Tags'] = df['Popular Tags'].apply(lambda x: x if isinstance(x, list) else [])
df['Game Features'] = df['Game Features'].apply(lambda x: x if isinstance(x, list) else [])

# Remove rows with critical missing data
df = df.dropna(subset=['Title', 'Game Description'])
print("Data shape after cleaning:", df.shape)

# 2. Enhanced feature engineering
def create_enhanced_features(row):
    try:
        # Safely handle tags and features
        tags = ' '.join([str(tag).lower() for tag in row['Popular Tags']])
        features = ' '.join([str(feat).lower() for feat in row['Game Features']])
        description = re.sub(r'[^\w\s]', '', str(row['Game Description']).lower())
        title = str(row['Title']).lower()
        return f"{title} {title} {tags} {tags} {features} {description}"
    except Exception as e:
        print(f"Error processing row {row.name}: {e}")
        return ""

df['enhanced_features'] = df.apply(create_enhanced_features, axis=1)

# 3. Create the ML pipeline
pipeline = make_pipeline(
    TfidfVectorizer(
        stop_words='english',
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2
    ),
    Normalizer()
)

sparse_matrix = pipeline.fit_transform(df['enhanced_features'])

knn = NearestNeighbors(
    n_neighbors=6,
    metric='cosine',
    algorithm='auto',
    n_jobs=-1
)
knn.fit(sparse_matrix)

# 4. Robust recommendation function
def enhanced_recommend(title):
    try:
        # Find matches safely
        mask = df['Title'].notna() & df['Title'].str.contains(title, case=False, regex=False)
        matches = df[mask]
        
        if len(matches) == 0:
            print(f"\n⚠️ No games found matching: '{title}'")
            print("Try these popular games instead:")
            return df[['Title', 'Link']].head(5)
        
        idx = matches.index[0]
        query = pipeline.transform([df.loc[idx, 'enhanced_features']])
        
        distances, indices = knn.kneighbors(query, n_neighbors=6)
        
        results = []
        for i, distance in zip(indices[0][1:], distances[0][1:]):  # Skip self
            game = df.iloc[i]
            results.append({
                'Title': game['Title'],
                'Link': game['Link'],
                'Similarity': f"{1 - distance:.3f}",
                'Price': game.get('Discounted Price', game.get('Original Price', 'N/A')),
                'Genres': ', '.join(map(str, game['Popular Tags'][:3])) if game['Popular Tags'] else ''
            })
        
        return pd.DataFrame(results)
    
    except Exception as e:
        print(f"\n❌ Error processing '{title}': {str(e)}")
        return pd.DataFrame(columns=['Title', 'Link', 'Similarity', 'Price', 'Genres'])

# 5. Test the system
print("\nFirst 5 games in dataset:")
print(df[['Title']].head().to_markdown(index=False))

# test_game = input("\nEnter a game title to test (or press Enter for default): ") or "Baldur's Gate 3"
# recommendations = enhanced_recommend(test_game)

# if not recommendations.empty:
#     print("\nRecommended Games:")
#     print(recommendations[['Title', 'Similarity', 'Price']].to_markdown(index=False))
# else:
#     print("\nNo recommendations could be generated.")
    
import joblib
from datetime import datetime

# ... [your existing code until after knn.fit(sparse_matrix)] ...

# Save the model components
model_components = {
    'pipeline': pipeline,
    'knn': knn,
    'df': df,  # Save the dataframe for reference
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

# Save to file
joblib.dump(model_components, 'game_recommender_model.joblib')

print("Model saved successfully as 'game_recommender_model.joblib'")

# ... [rest of your existing code] ...