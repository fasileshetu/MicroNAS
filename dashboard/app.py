import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import ast

st.set_page_config(page_title="MicroNAS Dashboard", layout="wide")
st.title("MicroNAS — Neural Architecture Search")
st.markdown("Proxy-guided architecture search on the Credit Card Fraud Detection dataset.")

@st.cache_data
def load_phase(path):
    df = pd.read_csv(path)
    df['layers'] = df['layers'].apply(ast.literal_eval)
    df['activations'] = df['activations'].apply(ast.literal_eval)
    df['layers_str'] = df['layers'].apply(str)
    df['activations_str'] = df['activations'].apply(str)
    df['eval_order'] = range(1, len(df) + 1)
    return df

@st.cache_data
def load_forward_selection():
    with open('results/forward_selection.json') as f:
        return json.load(f)

try:
    phase1 = load_phase('results/phase1_diversity.csv')
    phase2 = load_phase('results/phase2_diversity_rf.csv')
    fs_data = load_forward_selection()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False
    st.error("Results not found. Run the pipeline first: python main.py")

if data_loaded:

    # section 1 — benchmark summary
    st.header("Benchmark Summary")
    col1, col2, col3, col4 = st.columns(4)

    p1_best = phase1.loc[phase1['val_score'].idxmax()]
    p2_best = phase2.loc[phase2['val_score'].idxmax()]
    gap = p2_best['val_score'] - p1_best['val_score']

    with col1:
        st.metric(
            label="Phase 1 Best (Diversity Heuristic)",
            value=f"{p1_best['val_score']:.4f} AUC-PR",
            delta=None
        )
        st.caption(f"Architecture: {p1_best['layers_str']}")
        st.caption(f"Params: {p1_best['param_count']:,}")

    with col2:
        st.metric(
            label="Phase 2 Best (RF Proxy Guided)",
            value=f"{p2_best['val_score']:.4f} AUC-PR",
            delta=f"{gap:+.4f} vs Phase 1"
        )
        st.caption(f"Architecture: {p2_best['layers_str']}")
        st.caption(f"Params: {p2_best['param_count']:,}")

    with col3:
        st.metric(
            label="Proxy Quality (Kendall's Tau)",
            value="0.4595",
            delta="+0.0587 vs budget=50"
        )
        st.caption("Top-10 Overlap: 40%")
        st.caption("Trained on 150 architectures")

    with col4:
        st.metric(
            label="Post-NAS: 30 vs 15 Features",
            value="0.8407 vs 0.6938",
            delta="+0.1469 (30 features wins)"
        )
        st.caption("Architecture: [128] relu, 3 runs each")
        st.caption("Forward selection hurts final model quality")

    st.divider()

    # section 2 — AUC-PR across evaluations
    st.header("AUC-PR Across Evaluations")

    phase1['Phase'] = 'Phase 1 — Diversity Heuristic'
    phase2['Phase'] = 'Phase 2 — RF Proxy Guided'
    combined = pd.concat([phase1, phase2], ignore_index=True)

    fig = px.scatter(
        combined,
        x='eval_order',
        y='val_score',
        color='Phase',
        hover_data=['layers_str', 'activations_str', 'param_count'],
        labels={'eval_order': 'Evaluation Number', 'val_score': 'AUC-PR'},
        title='AUC-PR Score per Evaluation (150 architectures per phase)'
    )

    fig.add_hline(
        y=p1_best['val_score'],
        line_dash='dash',
        line_color='blue',
        annotation_text=f"Phase 1 best: {p1_best['val_score']:.4f}"
    )
    fig.add_hline(
        y=p2_best['val_score'],
        line_dash='dash',
        line_color='red',
        annotation_text=f"Phase 2 best: {p2_best['val_score']:.4f}"
    )

    st.plotly_chart(fig, config={'displayModeBar': True})

    st.divider()

    # section 3 — feature importance
    st.header("Forward Selection — Feature Importance")
    st.markdown("Features selected in order of predictive value. Earlier = more important.")
    st.markdown("Note: post-NAS comparison shows 30 features outperforms 15 selected features by 0.1469 AUC-PR on the final model.")

    feature_df = pd.DataFrame({
        'Feature': fs_data['names'],
        'Selection Order': range(1, len(fs_data['names']) + 1),
        'Importance Score': [1 / i for i in range(1, len(fs_data['names']) + 1)]
    })

    fig2 = px.bar(
        feature_df,
        x='Feature',
        y='Importance Score',
        title='Feature Importance by Selection Order',
        labels={'Importance Score': 'Relative Importance (1/rank)'}
    )
    st.plotly_chart(fig2, config={'displayModeBar': True})

    st.divider()

    # section 4 — architecture explorer
    st.header("Architecture Explorer")

    phase_filter = st.selectbox("Select Phase", ["Both", "Phase 1", "Phase 2"])
    sort_by = st.selectbox("Sort By", ["val_score", "param_count", "train_time"])
    ascending = st.checkbox("Ascending", value=False)

    if phase_filter == "Phase 1":
        explorer_df = phase1.copy()
    elif phase_filter == "Phase 2":
        explorer_df = phase2.copy()
    else:
        explorer_df = combined.copy()

    explorer_df = explorer_df.sort_values(sort_by, ascending=ascending)

    st.dataframe(
        explorer_df[['Phase', 'layers_str', 'activations_str', 'val_score', 'param_count', 'train_time']].rename(columns={
            'layers_str': 'Layers',
            'activations_str': 'Activations',
            'val_score': 'AUC-PR',
            'param_count': 'Params',
            'train_time': 'Train Time (s)',
            'Phase': 'Phase'
        }),
        hide_index=True
    )