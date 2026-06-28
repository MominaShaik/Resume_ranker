"""
Premium Streamlit Dashboard for Redrob Candidate Ranking System.
Displays ranked candidates with interactive visualizations and deep-dive profile inspection.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


# Page configuration
st.set_page_config(
    page_title="Redrob AI Candidate Ranking",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium dark mode styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    
    .stApp {
        background-color: #0e1117;
    }
    
    /* Metric cards with glowing effect */
    .metric-card {
        background: linear-gradient(135deg, #1a1d24 0%, #252830 100%);
        border: 1px solid #3d4450;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 6px 30px rgba(0, 150, 255, 0.2);
        border-color: #00a8ff;
    }
    
    /* Custom container for candidate cards */
    .candidate-card {
        background: linear-gradient(135deg, #1a1d24 0%, #252830 100%);
        border: 1px solid #3d4450;
        border-radius: 12px;
        padding: 24px;
        margin: 12px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Header styling */
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00a8ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Subheader styling */
    .subheader {
        color: #8892b0;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #00a8ff;
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        background-color: #1a1d24;
        border: 1px solid #3d4450;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background-color: #1a1d24;
        border-radius: 8px;
    }
    
    /* Reasoning text styling */
    .reasoning-text {
        color: #ccd6f6;
        font-style: italic;
        line-height: 1.6;
    }
    
    /* Score badge styling */
    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .score-high {
        background: linear-gradient(135deg, #00ff88, #00cc6a);
        color: #0a0a0a;
    }
    
    .score-medium {
        background: linear-gradient(135deg, #00a8ff, #0088cc);
        color: #ffffff;
    }
    
    .score-low {
        background: linear-gradient(135deg, #ff6b6b, #cc5555);
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


def load_ranked_data():
    """Load the ranked candidates CSV file."""
    csv_path = Path("output/ranked_candidates.csv")
    
    if not csv_path.exists():
        st.error("❌ No ranking results found. Please run `python main.py` first to generate rankings.")
        return None
    
    df = pd.read_csv(csv_path)
    return df


def render_metric_card(title: str, value: str, subtitle: str, color: str = "#00a8ff"):
    """Render a premium metric card with glowing effect."""
    st.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid {color};">
        <h3 style="color: #8892b0; font-size: 0.9rem; margin: 0; text-transform: uppercase; letter-spacing: 1px;">{title}</h3>
        <div style="font-size: 2.5rem; font-weight: 700; color: {color}; margin: 10px 0;">{value}</div>
        <div style="color: #64ffda; font-size: 0.85rem;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def get_score_tier(score: float) -> tuple:
    """Determine score tier and color."""
    if score >= 0.7:
        return "Elite", "#00ff88"
    elif score >= 0.5:
        return "Strong", "#00a8ff"
    elif score >= 0.3:
        return "Moderate", "#ffd700"
    else:
        return "Low", "#ff6b6b"


def render_score_progress(score: float):
    """Render an interactive progress bar for the score."""
    percentage = min(score * 100, 100)
    
    # Determine color based on score
    if score >= 0.7:
        color = "#00ff88"
    elif score >= 0.5:
        color = "#00a8ff"
    elif score >= 0.3:
        color = "#ffd700"
    else:
        color = "#ff6b6b"
    
    st.markdown(f"""
    <div style="margin: 8px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="color: #8892b0; font-size: 0.85rem;">AI Match Score</span>
            <span style="color: {color}; font-weight: 600; font-size: 0.85rem;">{score:.4f}</span>
        </div>
        <div style="background: #1a1d24; border-radius: 6px; height: 8px; overflow: hidden;">
            <div style="background: {color}; height: 100%; width: {percentage}%; transition: width 0.5s ease;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_candidate_card(row: pd.Series, rank: int):
    """Render a detailed candidate card."""
    score = row['score']
    tier, color = get_score_tier(score)
    
    st.markdown(f"""
    <div class="candidate-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div>
                <span style="color: #00a8ff; font-weight: 700; font-size: 1.1rem;">#{rank}</span>
                <span style="color: #ccd6f6; font-weight: 600; font-size: 1.1rem; margin-left: 12px;">{row['candidate_id']}</span>
            </div>
            <span class="score-badge score-{tier.lower()}" style="background: {color};">{tier} Match</span>
        </div>
        {render_score_progress(score)}
        <div class="reasoning-text" style="margin-top: 12px;">
            <strong style="color: #64ffda;">AI Analysis:</strong> {row['reasoning']}
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="header-title">🎯 Redrob AI Candidate Ranking</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Intelligent Candidate Discovery & Ranking System</p>', unsafe_allow_html=True)
    
    # Load data
    df = load_ranked_data()
    
    if df is None:
        st.stop()
    
    # Calculate metrics
    total_processed = 100000  # As per hackathon requirements
    total_shortlisted = len(df)
    avg_score = df['score'].mean()
    top_score = df['score'].max()
    
    # Determine match strength tier
    if avg_score >= 0.6:
        match_tier = "Elite"
        tier_color = "#00ff88"
    elif avg_score >= 0.4:
        match_tier = "Strong"
        tier_color = "#00a8ff"
    else:
        match_tier = "Moderate"
        tier_color = "#ffd700"
    
    # Render metric cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        render_metric_card(
            "Total Processed",
            f"{total_processed:,}",
            "Candidates analyzed via multistage pipeline",
            "#00a8ff"
        )
    
    with col2:
        render_metric_card(
            "Shortlisted Finalists",
            f"{total_shortlisted}",
            "Top candidates selected for review",
            "#00ff88"
        )
    
    with col3:
        render_metric_card(
            "Match Strength Tier",
            match_tier,
            f"Avg Score: {avg_score:.4f} | Top: {top_score:.4f}",
            tier_color
        )
    
    st.markdown("---")
    
    # Two-column layout
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        st.markdown('<h2 style="color: #ccd6f6; font-size: 1.5rem;">📊 Score Distribution</h2>', unsafe_allow_html=True)
        
        # Create Plotly histogram
        fig = px.histogram(
            df,
            x='score',
            nbins=20,
            title='Distribution of AI Match Scores',
            color_discrete_sequence=['#00a8ff']
        )
        
        fig.update_layout(
            plot_bgcolor='#1a1d24',
            paper_bgcolor='#0e1117',
            font=dict(color='#ccd6f6'),
            title_font=dict(color='#00a8ff', size=18),
            xaxis=dict(
                title='AI Match Score',
                gridcolor='#3d4450',
                tickcolor='#3d4450'
            ),
            yaxis=dict(
                title='Count',
                gridcolor='#3d4450',
                tickcolor='#3d4450'
            ),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        fig.update_traces(
            marker=dict(
                line=dict(color='#00a8ff', width=2),
                color='#00a8ff'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<h2 style="color: #ccd6f6; font-size: 1.5rem; margin-top: 30px;">🏆 Top Ranked Candidates</h2>', unsafe_allow_html=True)
        
        # Display top candidates with cards
        for idx, row in df.head(10).iterrows():
            render_candidate_card(row, idx + 1)
    
    with right_col:
        st.markdown('<h2 style="color: #ccd6f6; font-size: 1.5rem;">🔍 Profile Inspector</h2>', unsafe_allow_html=True)
        
        st.markdown('<p style="color: #8892b0; font-size: 0.9rem; margin-bottom: 15px;">Select a candidate to view detailed analysis</p>', unsafe_allow_html=True)
        
        # Dropdown for candidate selection
        candidate_options = [f"#{row['rank']} - {row['candidate_id']}" for _, row in df.iterrows()]
        selected_candidate = st.selectbox(
            "Select Candidate",
            candidate_options,
            label_visibility="collapsed"
        )
        
        if selected_candidate:
            # Parse selected candidate
            rank = int(selected_candidate.split('#')[1].split('-')[0].strip())
            candidate_id = selected_candidate.split('-')[1].strip()
            
            # Get candidate data
            candidate_data = df[df['candidate_id'] == candidate_id].iloc[0]
            
            # Render detailed card
            st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
            render_candidate_card(candidate_data, rank)
            
            # Additional metrics
            score = candidate_data['score']
            tier, color = get_score_tier(score)
            
            st.markdown(f"""
            <div style="margin-top: 20px; padding: 16px; background: #1a1d24; border-radius: 8px; border: 1px solid #3d4450;">
                <h4 style="color: #00a8ff; margin: 0 0 12px 0;">Performance Metrics</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                    <div>
                        <div style="color: #8892b0; font-size: 0.8rem;">Rank</div>
                        <div style="color: #ccd6f6; font-weight: 600; font-size: 1.1rem;">#{candidate_data['rank']}</div>
                    </div>
                    <div>
                        <div style="color: #8892b0; font-size: 0.8rem;">Score Tier</div>
                        <div style="color: {color}; font-weight: 600; font-size: 1.1rem;">{tier}</div>
                    </div>
                    <div>
                        <div style="color: #8892b0; font-size: 0.8rem;">Raw Score</div>
                        <div style="color: #ccd6f6; font-weight: 600; font-size: 1.1rem;">{score:.4f}</div>
                    </div>
                    <div>
                        <div style="color: #8892b0; font-size: 0.8rem;">Percentile</div>
                        <div style="color: #ccd6f6; font-weight: 600; font-size: 1.1rem;">{(1 - (candidate_data['rank'] - 1) / len(df)) * 100:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick stats
        st.markdown('<h3 style="color: #ccd6f6; font-size: 1.2rem; margin-top: 30px;">📈 Quick Stats</h3>', unsafe_allow_html=True)
        
        stats_col1, stats_col2 = st.columns(2)
        
        with stats_col1:
            st.metric("Elite Matches", len(df[df['score'] >= 0.7]), delta=f"{len(df[df['score'] >= 0.7]) / len(df) * 100:.1f}%")
        
        with stats_col2:
            st.metric("Strong Matches", len(df[df['score'] >= 0.5]), delta=f"{len(df[df['score'] >= 0.5]) / len(df) * 100:.1f}%")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #8892b0; font-size: 0.85rem; padding: 20px;">
        <p>🚀 Powered by Multistage AI Ranking Pipeline | Redrob Hackathon 2024</p>
        <p style="margin-top: 8px;">Stage 1: Lexical/BM25 Filter → Stage 2: Semantic Embeddings → Stage 3: Behavioral Scoring</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
