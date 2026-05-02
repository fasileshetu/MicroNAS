import streamlit as st
import pandas as pd
import plotly.express as px
import json
import ast

st.set_page_config(page_title="MicroNAS Dashboard", layout="wide")
st.title("MicroNAS — Neural Architecture Search")
st.markdown("Proxy-guided A* search for efficient neural architecture discovery on the ULB Credit Card Fraud Detection dataset (284,807 transactions, ~0.17% fraud rate).")

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

@st.cache_data
def load_random_search():
    df = pd.read_csv('results/random_search.csv')
    df['layers_str'] = df['layers'].apply(str)
    return df

@st.cache_data
def load_successive_halving():
    df = pd.read_csv('results/successive_halving.csv')
    df['layers_str'] = df['layers'].apply(str)
    return df

try:
    phase1 = load_phase('results/phase1_diversity.csv')
    phase2 = load_phase('results/phase2_diversity_rf.csv')
    fs_data = load_forward_selection()
    rs_data = load_random_search()
    sh_data = load_successive_halving()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False
    st.error("Results not found. Run the pipeline first: python main.py")

if data_loaded:

    # section 1 — benchmark summary
    st.header("Benchmark Summary")
    col1, col2, col3, col4, col5 = st.columns(5)

    p1_best = phase1.loc[phase1['val_score'].idxmax()]
    p2_best = phase2.loc[phase2['val_score'].idxmax()]
    gap = p2_best['val_score'] - p1_best['val_score']
    rs_best = rs_data['val_score'].max()
    sh_best = sh_data[sh_data['epochs'] == 10]['val_score'].max()

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
            label="vs Random Search",
            value=f"{rs_best:.4f} AUC-PR",
            delta=f"{p2_best['val_score'] - rs_best:+.4f} MicroNAS advantage"
        )
        st.caption("Random search best: [512] relu")
        st.caption("15,872 params — 4x less efficient")

    with col4:
        st.metric(
            label="Proxy Quality (Kendall's Tau)",
            value="0.4595",
            delta="+0.0587 vs budget=50"
        )
        st.caption("Top-10 Overlap: 40%")
        st.caption("Trained on 150 architectures")

    with col5:
        st.metric(
            label="Post-NAS: 30 vs 15 Features",
            value="0.8407 vs 0.6938",
            delta="+0.1469 (30 features wins)"
        )
        st.caption("[128] relu, 3 runs each averaged")
        st.caption("Forward selection hurts model quality")

    st.divider()

    # section 2 — method comparison
    st.header("Method Comparison")

    comparison_df = pd.DataFrame({
        'Method': ['MicroNAS Phase 2', 'SuccessiveHalving (budget=50)', 'Random Search', 'SuccessiveHalving (budget=150)'],
        'Best AUC-PR': [0.8676, 0.8590, 0.8558, 0.8398],
        'Total Evaluations': [150, 93, 150, 280],
        'Best Architecture': ['[128] relu', '[128] relu', '[512] relu', '[64] relu'],
        'Params': [3968, 3968, 15872, 1984]
    })

    fig0 = px.bar(
        comparison_df,
        x='Method',
        y='Best AUC-PR',
        color='Method',
        text='Best AUC-PR',
        title='AUC-PR by Search Method',
        labels={'Best AUC-PR': 'AUC-PR'}
    )
    fig0.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig0.update_layout(showlegend=False, yaxis_range=[0.82, 0.875])
    st.plotly_chart(fig0, config={'displayModeBar': True})

    st.divider()

    # section 3 — AUC-PR across evaluations
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

    # section 4 — proxy quality
    st.header("Proxy Quality")
    st.markdown("Kendall's Tau measures how well the surrogate model ranks architectures by predicted AUC-PR vs actual AUC-PR. Higher = better proxy.")

    proxy_df = pd.DataFrame({
        'Budget': ['budget=50', 'budget=150'],
        "Kendall's Tau": [0.4008, 0.4595],
        'Top-10 Overlap': [0.70, 0.40]
    })

    col_a, col_b = st.columns(2)
    with col_a:
        fig_tau = px.bar(
            proxy_df,
            x='Budget',
            y="Kendall's Tau",
            text="Kendall's Tau",
            title="Kendall's Tau by Budget",
            color='Budget'
        )
        fig_tau.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig_tau.update_layout(showlegend=False, yaxis_range=[0, 0.6])
        st.plotly_chart(fig_tau, config={'displayModeBar': True})

    with col_b:
        fig_topk = px.bar(
            proxy_df,
            x='Budget',
            y='Top-10 Overlap',
            text='Top-10 Overlap',
            title='Top-10 Overlap by Budget',
            color='Budget'
        )
        fig_topk.update_traces(texttemplate='%{text:.0%}', textposition='outside')
        fig_topk.update_layout(showlegend=False, yaxis_range=[0, 1])
        st.plotly_chart(fig_topk, config={'displayModeBar': True})

    st.divider()

    # section 5 — feature importance
    st.header("Forward Selection — Feature Importance")
    st.markdown("Features selected in order of predictive value by logistic regression forward selection.")
    st.markdown("Note: post-NAS comparison shows 30 features outperforms 15 selected features by **0.1469 AUC-PR** — forward selection actively hurts final model quality on this dataset.")

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

    # section 6 — architecture explorer
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