# MicroNAS
I implemented a micro-scale NAS (Neural Architecture Search) system from scratch — the same class of technique used by Google AutoML and DeepMind — that uses A* search with learned heuristics to discover efficient neural architectures, then benchmarks them against hand-designed baselines.

## Pipeline

Raw Data (30 features)
        ↓
Feature Selection (search/forward_selection.py)
  · Greedy forward selection using logistic regression
  · 3-fold cross validation to evaluate each candidate feature
  · Selects 15 of 30 most predictive features
  · Saves results to results/forward_selection.json
        ↓
Phase 1: Naive Architecture Search (search/astar.py)
  · A* search with naive heuristic (prefers smaller param counts)
  · Evaluates 50 architectures on selected features
  · Metric: AUC-PR (area under precision-recall curve)
  · Saves results incrementally to results/creditcard_phase1.csv
        ↓
Proxy Model Training (proxy/predictor.py)
  · Converts each architecture to a 26-feature numeric vector
  · Trains Ridge regression to predict AUC-PR without full training
  · StandardScaler applied for feature normalization
        ↓
Phase 2: Proxy Guided Architecture Search (search/astar.py)
  · Same A* search but heuristic replaced by learned proxy predictions
  · Evaluates 50 architectures guided toward promising regions
  · Saves results incrementally to results/creditcard_phase2.csv
        ↓
Benchmark Comparison
  · Phase 1 best: [64] relu at 0.8009 AUC-PR, 1,024 params
  · Phase 2 best: [64, 32] sigmoid/relu at 0.8277 AUC-PR, 3,040 params
  · Proxy guided search improved AUC-PR by 0.027