# MicroNAS
A micro-scale Neural Architecture Search (NAS) system built from scratch, applying A* search with learned proxy heuristics to discover efficient neural network architectures for credit card fraud detection.

The core idea: frame architecture design as a search problem. States are neural networks defined by hidden layer widths and activation functions. Operators mutate architectures. A learned surrogate model serves as the search heuristic, predicting AUC-PR without full training.

Best result: 0.8608 AUC-PR -- diversity heuristic Phase 1 + RandomForest proxy Phase 2.

----------------------------------------------------------------------------------------------------------------

## Pipeline
Raw Data (30 features, ULB Credit Card Fraud Detection)
        ↓
Phase 1: Architecture Search for Proxy Training Data
- A* search with a structural heuristic (no proxy yet)
- Evaluates 50 architectures, 10 epochs each
- Metric: AUC-PR (highly imbalanced dataset, ~0.17% fraud rate)
- Saves results incrementally to results/phase1_{heuristic}.csv
        ↓
Proxy Model Training (proxy/predictor.py)
- Converts each architecture to a 26-feature numeric vector
- Trains Ridge or RandomForest surrogate to predict AUC-PR
- StandardScaler applied for feature normalization
        ↓
Phase 2: Proxy-Guided Architecture Search
- Same A* search with heuristic replaced by learned proxy predictions
- Optional UCB exploration bonus: f = g - h - beta * uncertainty
- Evaluates 50 architectures guided toward high-AUC-PR regions
- Saves results incrementally to results/phase2_{config}.csv
        ↓
Analysis
- Compare Phase 1 vs Phase 2 best results
- Evaluate proxy type, heuristic, and UCB beta tradeoffs

----------------------------------------------------------------------------------------------------------------

## Key Findings
- 30 features outperforms 15 -- forward selection reduced score variance in Phase 1 data, degrading proxy signal. Reverted to raw 30 features for all proxy training.
- Phase 1 heuristic has limited impact with Ridge proxy -- all heuristics produced Phase 2 results in a narrow range (0.8407-0.8568).
- RandomForest proxy is more sensitive to Phase 1 data quality -- proxy choice and heuristic interact. Best result (0.8608) came from diversity heuristic + RF.
- Diversity heuristic consistently explores deeper architectures -- all diversity-aware heuristics on 30 features explored 3-layer architectures; naive never did.
- Removing exploration decay hurts proxy signal most -- produces the lowest score range across Phase 1 runs.

Full experiment history in EXPERIMENTS.md.

----------------------------------------------------------------------------------------------------------------

## Current Status
UCB beta sweep in progress: beta in {0.0, 0.1, 0.3, 0.5, 1.0} using diversity heuristic Phase 1 data + RandomForest proxy. Runs parallelized across betas with nohup. 

Next steps:
1. Analyze beta sweep results, pick best beta
2. Lock in final configuration
3. Post-NAS feature comparison: retrain best architecture on 30 vs 15 features, 3 runs each averaged
4. Update dashboard

----------------------------------------------------------------------------------------------------------------

## Project Structure
micronas/
├── search/
│   ├── space.py              # Architecture dataclass
│   ├── operators.py          # Mutation operators
│   ├── astar.py              # A* search loop
│   ├── heuristics.py         # Heuristic functions
│   └── forward_selection.py  # Greedy feature selection
├── train/
│   └── creditcard_trainer.py # Trainer with class weighting, AUC-PR, early stopping
├── proxy/
│   └── predictor.py          # Ridge/RF surrogate with StandardScaler
├── dashboard/
│   └── app.py                # Streamlit visualization
├── analysis/
│   └── phase1_analysis.py    # Diversity scoring for Phase 1 CSVs
├── tests/                    # 13 unit tests
├── results/                  # CSV outputs per experiment
├── EXPERIMENTS.md            # Full experiment history
└── main.py                   # Orchestration

----------------------------------------------------------------------------------------------------------------

## Dataset
ULB Credit Card Fraud Detection -- 284,807 transactions, 492 fraud (~0.17%). All features are PCA-transformed. AUC-PR used as metric due to severe class imbalance.
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud