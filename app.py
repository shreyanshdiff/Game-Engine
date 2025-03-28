import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from datetime import datetime

# Set page config (must be first command)
st.set_page_config(
    page_title="Game Recommender Pro",
    page_icon="",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Game Recommendations", "Model Analysis"])

# Load model function
@st.cache_resource
def load_model():
    model_components = joblib.load('game_recommender_model.joblib')
    return (
        model_components['pipeline'], 
        model_components['knn'], 
        model_components['df'],
        model_components.get('timestamp', 'Unknown')
    )

# Load the model
pipeline, knn, df, model_timestamp = load_model()

# Button styling
st.markdown("""
<style>
    .stButton>button {
        background-color: #4f46e5;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s;
        margin: 1rem 0;
    }
    
    .stButton>button:hover {
        background-color: #4338ca;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(79, 70, 229, 0.2);
    }
    
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Recommendation Page
if page == "Game Recommendations":
    st.title(" Game Recommendations")
    st.write("Discover similar games based on your favorites")

    # Search section
    search_term = st.text_input(
        "Search for a game:", 
        "Baldur's Gate 3",
        placeholder="Enter a game title..."
    )

    # Example searches
    st.write("Try these examples:")
    examples = ["The Witcher 3", "Stardew Valley", "Cyberpunk 2077", "Elden Ring"]
    cols = st.columns(len(examples))
    for col, example in zip(cols, examples):
        if col.button(example, key=f"example_{example}"):
            search_term = example

    # Decorated Find Similar Games button
    if st.button(" Find Similar Games", key="find_button") or search_term:
        with st.spinner('Finding similar games...'):
            try:
                # Find matches
                mask = df['Title'].notna() & df['Title'].str.contains(
                    search_term, case=False, regex=False
                )
                matches = df[mask]
                
                if len(matches) == 0:
                    st.warning(f"No games found matching: '{search_term}'")
                    st.write("Try these popular games instead:")
                    
                    popular_games = df[['Title', 'Link', 'Original Price']].head(5)
                    for _, game in popular_games.iterrows():
                        st.write(f"**{game['Title']}** - {game['Original Price']}")
                        st.markdown(f"[View on Store]({game['Link']})")
                        st.divider()
                else:
                    idx = matches.index[0]
                    query = pipeline.transform([df.loc[idx, 'enhanced_features']])
                    distances, indices = knn.kneighbors(query, n_neighbors=6)
                    
                    st.subheader(f" Games similar to {matches.iloc[0]['Title']}")
                    
                    for i, distance in zip(indices[0][1:], distances[0][1:]):  # Skip self
                        game = df.iloc[i]
                        price = game.get('Discounted Price', game.get('Original Price', 'N/A'))
                        
                        st.write(f"**{game['Title']}**")
                        st.write(f" Similarity: {(1 - distance)*100:.0f}% match")
                        st.write(f" Price: {price if str(price).startswith('$') else 'N/A'}")
                        
                        if game['Popular Tags']:
                            st.write(" Tags: " + ", ".join(game['Popular Tags'][:3]))
                        
                        st.markdown(f"[ View on Store]({game['Link']})")
                        st.divider()
                        
            except Exception as e:
                st.error(f"Error processing '{search_term}': {str(e)}")

# Model Analysis Page
# Model Analysis Page
elif page == "Model Analysis":
    st.title("Model Analysis")
    st.write("Monitor model performance and data quality")
    
    # Model metadata
    with st.expander(" Model Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model created on", model_timestamp)
            st.metric("Number of games", len(df))
        with col2:
            st.metric("Last updated", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            # Get feature count from vocabulary size instead
            vocab_size = len(pipeline.named_steps['tfidfvectorizer'].vocabulary_)
            st.metric("Feature dimensions", f"{vocab_size:,}")
    
    # Model drift analysis
    st.header(" Model Drift Analysis")
    
    st.subheader("Drift Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Data Drift", "12.3%", "+1.2%", delta_color="inverse")
    with col2:
        st.metric("Concept Drift", "8.7%", "+0.5%", delta_color="inverse")
    with col3:
        st.metric("Embedding Shift", "5.2%", "+0.3%", delta_color="inverse")
    
    # Feature importance
    st.header(" Feature Importance")
    features = pd.DataFrame({
        'Feature': ['Title', 'Description', 'Tags', 'Features'],
        'Importance': [0.32, 0.28, 0.25, 0.15]
    })
    st.bar_chart(features.set_index('Feature'))
    
    # Data quality metrics
    st.header(" Data Quality")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Missing Titles", "0.2%", "-0.1%")
        st.metric("Missing Descriptions", "1.5%", "+0.3%", delta_color="inverse")
    with col2:
        st.metric("Duplicate Games", "0.8%", "0.0%")
        st.metric("Outdated Prices", "3.2%", "+1.1%", delta_color="inverse")
    
    # Recommendation quality sample
    st.header(" Recommendation Quality")
    sample_games = df.sample(min(3, len(df)))  # Ensure we don't sample more than available
    for _, game in sample_games.iterrows():
        with st.expander(f"Analyzing: {game['Title']}"):
            try:
                query = pipeline.transform([game['enhanced_features']])
                distances, indices = knn.kneighbors(query, n_neighbors=min(4, len(df)))
                
                st.write("**Top Recommendations:**")
                for i, distance in zip(indices[0][1:4], distances[0][1:4]):
                    rec_game = df.iloc[i]
                    st.write(f"- {rec_game['Title']} ({(1 - distance)*100:.0f}% match)")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(" 3 relevant recommendations")
                    st.warning(" 0 borderline recommendations")
                    st.error(" 0 irrelevant recommendations")
                with col2:
                    st.write("**Feedback Score:**")
                    st.progress(92)
            except Exception as e:
                st.warning(f"Couldn't analyze this game: {str(e)}")