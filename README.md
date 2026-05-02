# MicroNAS

A micro-scale Neural Architecture Search (NAS) system built from scratch, applying A* search with learned proxy heuristics to discover efficient neural network architectures for credit card fraud detection.

The core idea: frame architecture design as a search problem. States are neural networks defined by hidden layer widths and activation functions. Operators mutate architectures. A learned surrogate model serves as the search heuristic, predicting AUC-PR without full training.

Best result: 0.8676 AUC-PR -- diversity heuristic Phase 1 + RandomForest proxy Phase 2, budget=150.

---

## Pipeline

    Raw Data (30 features, ULB Credit Card Fraud Detection)
            |
    Phase 1: Architecture Search for Proxy Training Data
      - A* search with diversity heuristic (no proxy yet)
      - Evaluates 150 architectures, 10 epochs each
      - Metric: AUC-PR (highly imbalanced dataset, ~0.17% fraud rate)
      - Saves results incrementally to results/phase1_diversity.csv
            |
    Proxy Model Training (proxy/predictor.py)
      - Converts each architecture to a 26-feature numeric vector
      - Trains RandomForest surrogate to predict AUC-PR
      - StandardScaler applied for feature normalization
            |
    Phase 2: Proxy-Guided Architecture Search
      - Same A* search with heuristic replaced by learned proxy predictions
      - Evaluates 150 architectures guided toward high-AUC-PR regions
      - Saves results incrementally to results/phase2_diversity_rf.csv
            |
    Analysis
      - Compare Phase 1 vs Phase 2 best results
      - Proxy quality: Kendall's Tau, Top-10 overlap
      - Baseline comparisons: random search, SuccessiveHalving

---

## Key Findings

- 30 features outperforms 15 -- forward selection reduced score variance in Phase 1
  data, degrading proxy signal. Reverted to raw 30 features for all proxy training.
- Post-NAS feature comparison confirms this -- [128] relu scores 0.8407 on 30 features
  vs 0.6938 on 15 features (-0.1469 gap).
- Phase 1 heuristic has limited impact with Ridge proxy -- all heuristics produced
  Phase 2 results in a narrow range (0.8407-0.8568).
- RandomForest proxy is more sensitive to Phase 1 data quality -- best result (0.8676)
  came from diversity heuristic + RF at budget=150.
- Diversity heuristic consistently explores deeper architectures -- all diversity-aware
  heuristics on 30 features explored 3-layer architectures; naive never did.
- MicroNAS outperforms random search (+0.0118 AUC-PR) while finding a 4x more
  parameter-efficient architecture ([128] relu, 3,968 params vs [512] relu, 15,872 params).
- SuccessiveHalving at budget=50 approaches MicroNAS (0.8590) using less compute,
  but does not scale to budget=150 -- low-fidelity scores are too noisy on this dataset.
- Proxy quality: Kendall's Tau 0.4595, Top-10 overlap 40% (budget=150).

Full experiment history in EXPERIMENTS.md.

---

## Project Structure

    micronas/
    ├── search/
    │   ├── space.py              # Architecture dataclass
    │   ├── operators.py          # Mutation operators
    │   ├── astar.py              # A* search loop with timestamps
    │   ├── heuristics.py         # Heuristic functions
    │   └── forward_selection.py  # Greedy feature selection (results preserved)
    ├── train/
    │   └── creditcard_trainer.py # Trainer with class weighting, AUC-PR, early stopping
    ├── proxy/
    │   └── predictor.py          # Ridge/RF surrogate with StandardScaler
    ├── dashboard/
    │   └── app.py                # Streamlit visualization
    ├── analysis/
    │   ├── phase1_analysis.py    # Phase 1 diversity scoring
    │   ├── proxy_quality.py      # Kendall's Tau and Top-10 overlap
    │   ├── feature_comparison.py # Post-NAS 30 vs 15 feature comparison
    │   ├── random_search.py      # Random search baseline
    │   └── successive_halving.py # Multi-fidelity SuccessiveHalving
    ├── tests/                    # 13 unit tests
    ├── results/                  # CSV outputs per experiment
    ├── run_beta.py               # UCB beta sweep runner
    ├── EXPERIMENTS.md            # Full experiment history
    └── main.py                   # Orchestration (Phase 1 + Phase 2, budget=150)

---

## References

White, C., Safari, M., Sukthanker, R., Ru, B., Elsken, T., Zela, A., Dey, D., & Hutter, F.
Neural Architecture Search: Insights from 1000 Papers. arXiv:2301.08727, 2023.
https://arxiv.org/pdf/2301.08727

Read as part of this project to understand NAS search spaces, black-box optimization,
performance prediction, and multi-fidelity methods. Informed the SuccessiveHalving
implementation and proxy surrogate design.

---

## Dataset

ULB Credit Card Fraud Detection -- 284,807 transactions, 492 fraud (~0.17%). All
features are PCA-transformed. AUC-PR used as metric due to severe class imbalance.
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud