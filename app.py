import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors

# --- Load Model ---
@st.cache_resource
def load_model():
    model_components = joblib.load('game_recommender_model.joblib')
    return (
        model_components['pipeline'], 
        model_components['knn'], 
        model_components['df']
    )

pipeline, knn, df = load_model()

# --- Custom CSS for shadcn-like UI ---
def inject_custom_css():
    st.markdown("""
    <style>
    :root {
        --background: hsl(0 0% 100%);
        --foreground: hsl(222.2 84% 4.9%);
        --primary: hsl(221.2 83.2% 53.3%);
        --primary-foreground: hsl(210 40% 98%);
        --border: hsl(214.3 31.8% 91.4%);
        --input: hsl(214.3 31.8% 91.4%);
        --ring: hsl(221.2 83.2% 53.3%);
        --radius: 0.5rem;
    }
    
    .dark {
        --background: hsl(222.2 84% 4.9%);
        --foreground: hsl(210 40% 98%);
    }
    
    /* Main container */
    .main {
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Card styling */
    .card {
        background: var(--background);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);
    }
    
    /* Button styling */
    .btn {
        background-color: var(--primary);
        color: var(--primary-foreground);
        border-radius: var(--radius);
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .btn:hover {
        background-color: hsl(221.2 83.2% 43.3%);
    }
    
    /* Input styling */
    .input {
        display: flex;
        width: 100%;
        border-radius: var(--radius);
        border: 1px solid var(--border);
        background: var(--background);
        padding: 0.5rem 0.75rem;
        font-size: 0.875rem;
        line-height: 1.25rem;
        transition: border-color 0.2s;
    }
    
    .input:focus {
        outline: none;
        border-color: var(--ring);
    }
    
    /* Badge styling */
    .badge {
        display: inline-flex;
        align-items: center;
        border-radius: 9999px;
        background-color: hsl(221.2 83.2% 93.3%);
        color: hsl(221.2 83.2% 33.3%);
        padding: 0.25rem 0.75rem;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    /* Game card */
    .game-card {
        display: flex;
        gap: 1rem;
        padding: 1rem;
        border: 1px solid var(--border);
        border-radius: var(--radius);
        margin-bottom: 1rem;
        transition: all 0.2s;
    }
    
    .game-card:hover {
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    
    .game-info {
        flex: 1;
    }
    
    .game-title {
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    
    .game-meta {
        display: flex;
        gap: 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.875rem;
        color: hsl(215.4 16.3% 46.9%);
    }
    
    .game-tags {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.5rem;
    }
    
    .game-tag {
        background-color: hsl(210 40% 96.1%);
        color: hsl(215.4 16.3% 46.9%);
        padding: 0.25rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.75rem;
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- Header Section ---
st.markdown("""
<div class="main">
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-size: 2rem; font-weight: 800; margin-bottom: 0.5rem;">GameFinder</h1>
        <p style="color: hsl(215.4 16.3% 46.9%); max-width: 600px; margin: 0 auto;">
            Discover your next favorite game with AI-powered recommendations
        </p>
    </div>
""", unsafe_allow_html=True)

# --- Search Section ---
with st.container():
    st.markdown("""
    <div class="card">
        <h3 style="margin-top: 0; margin-bottom: 1rem;">Find Similar Games</h3>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input(
            "Search for a game:", 
            "Baldur's Gate 3", 
            key="search",
            label_visibility="collapsed",
            placeholder="Enter a game title..."
        )
    with col2:
        st.markdown("""
        <div style="display: flex; align-items: flex-end; height: 100%;">
            <button class="btn" id="search-btn">Search</button>
        </div>
        <script>
            document.getElementById("search-btn").addEventListener("click", function() {
                window.parent.document.querySelector('.stButton button').click();
            });
        </script>
        """, unsafe_allow_html=True)
    
    search_clicked = st.button("Search", key="search_btn", help="Click to search")
    
    st.markdown("</div>", unsafe_allow_html=True)

if search_clicked or search_term:
    with st.spinner('Finding similar games...'):
        try:
            # Find matches
            mask = df['Title'].notna() & df['Title'].str.contains(
                search_term, case=False, regex=False
            )
            matches = df[mask]
            
            if len(matches) == 0:
                st.markdown("""
                <div class="card">
                    <div style="color: hsl(0 72.2% 50.6%); margin-bottom: 0.5rem;">
                        ⚠️ No games found matching: '{}'
                    </div>
                    <p>Try these popular games instead:</p>
                </div>
                """.format(search_term), unsafe_allow_html=True)
                
                popular_games = df[['Title', 'Link']].head(5)
                for _, game in popular_games.iterrows():
                    st.markdown(f"""
                    <div class="game-card">
                        <div class="game-info">
                            <div class="game-title">{game['Title']}</div>
                            <a href="{game['Link']}" target="_blank" style="font-size: 0.875rem; color: var(--primary); text-decoration: none;">View on Store →</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                idx = matches.index[0]
                query = pipeline.transform([df.loc[idx, 'enhanced_features']])
                distances, indices = knn.kneighbors(query, n_neighbors=6)
                
                st.markdown(f"""
                <div class="card">
                    <h3 style="margin-top: 0;">Games similar to <span style="color: var(--primary);">{matches.iloc[0]['Title']}</span></h3>
                </div>
                """, unsafe_allow_html=True)
                
                for i, distance in zip(indices[0][1:], distances[0][1:]):  # Skip self
                    game = df.iloc[i]
                    
                    # Get price (try discounted first, then original)
                    price = game.get('Discounted Price', game.get('Original Price', 'N/A'))
                    if str(price).startswith('$'):
                        price_badge = f"""<span class="badge">{price}</span>"""
                    else:
                        price_badge = ""
                    
                    # Format tags
                    if game['Popular Tags']:
                        tags_html = "".join([
                            f"""<span class="game-tag">{tag}</span>""" 
                            for tag in game['Popular Tags'][:3]
                        ])
                    else:
                        tags_html = ""
                    
                    st.markdown(f"""
                    <div class="game-card">
                        <div class="game-info">
                            <div class="game-title">{game['Title']}</div>
                            <div class="game-meta">
                                <span>{(1 - distance)*100:.0f}% match</span>
                                {price_badge}
                            </div>
                            <div class="game-tags">
                                {tags_html}
                            </div>
                        </div>
                        <a href="{game['Link']}" target="_blank" style="text-decoration: none;">
                            <button class="btn" style="align-self: center; cursor: pointer; margin: auto 0;">View</button>
                        </a>
                    </div>
                    """, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"Error processing '{search_term}': {str(e)}")

# --- Examples Section ---
st.markdown("""
<div class="card">
    <h3 style="margin-top: 0;">Try these examples</h3>
    <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
        <button class="btn" onclick="document.querySelector('input[aria-label=\\'Search for a game:\\']').value='The Witcher 3'; document.querySelector('.stButton button').click()" style="font-size: 0.875rem; padding: 0.25rem 0.5rem;">The Witcher 3</button>
        <button class="btn" onclick="document.querySelector('input[aria-label=\\'Search for a game:\\']').value='Stardew Valley'; document.querySelector('.stButton button').click()" style="font-size: 0.875rem; padding: 0.25rem 0.5rem;">Stardew Valley</button>
        <button class="btn" onclick="document.querySelector('input[aria-label=\\'Search for a game:\\']').value='Cyberpunk 2077'; document.querySelector('.stButton button').click()" style="font-size: 0.875rem; padding: 0.25rem 0.5rem;">Cyberpunk 2077</button>
        <button class="btn" onclick="document.querySelector('input[aria-label=\\'Search for a game:\\']').value='Elden Ring'; document.querySelector('.stButton button').click()" style="font-size: 0.875rem; padding: 0.25rem 0.5rem;">Elden Ring</button>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div style="text-align: center; margin-top: 3rem; color: hsl(215.4 16.3% 46.9%); font-size: 0.875rem;">
    <p>Recommendations are based on game metadata similarity</p>
</div>
</div>
""", unsafe_allow_html=True)