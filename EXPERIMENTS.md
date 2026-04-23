# Experiments

All experiments run on [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB).
Metric: AUC-PR (area under precision-recall curve).
Budget: 50 evaluations per phase, 10 epochs per evaluation.

## Results

| Run                                      | Features | Heuristic    | Proxy              | Phase 1 | Phase 2 | Gap    |
|------------------------------------------|----------|--------------|--------------------|---------|---------|--------|
| MNIST baseline                           | 784      | Naive        | Ridge              | 91.8%   | 92.6%   | +0.8%  |
| Sampled 50k, naive + Ridge               | 30       | Naive        | Ridge              | 0.9188  | 0.9188  | +0.000 |
| naive + Ridge                            | 30       | Naive        | Ridge              | 0.7978  | 0.8145  | +0.017 |
| naive + Ridge                            | 15       | Naive        | Ridge              | 0.8273  | 0.8105  | -0.017 |
| naive + RF                               | 15       | Naive        | RandomForest       | 0.8015  | 0.8174  | +0.016 |
| diversity + Ridge                        | 15       | Diversity    | Ridge              | 0.8009  | 0.8318  | +0.032 |
| diversity + RF + UCB (beta=0.5)          | 15       | Diversity    | RF + UCB           | 0.7997  | 0.8279  | +0.028 |
| diversity + RF + UCB (beta=1.5)          | 15       | Diversity    | RF + UCB           | 0.7997  | 0.7969  | -0.003 |

## Phase 1 Heuristic Ablation (15 features, Round 1)

| Heuristic                  | Score Range | Layer Configs | Act Combos | Max Depth | Diversity Score |
|----------------------------|-------------|---------------|------------|-----------|-----------------|
| Naive (baseline)           | 0.213       | 9             | 12         | 2         | 0.357           |
| A — equal weights          | 0.226       | 9             | 12         | 2         | 0.370           |
| B — no depth               | 0.151       | 9             | 12         | 2         | 0.295           |
| C — no activation diversity| 0.217       | 9             | 12         | 2         | 0.361           |
| D — no exploration decay   | 0.188       | 9             | 12         | 2         | 0.332           |
| E — no size score          | 0.227       | 12            | 16         | 3         | 0.419           |
| Diversity v2               | 0.197       | 9             | 12         | 2         | 0.341           |

## Key Findings

- Diversity heuristic outperformed naive on 15 features (+0032 gap vs +0.017)
- 15 features hurt proxy quality vs 30 features — cleaner data means less score variance, less signal for proxy to learn from
- Heuristic E (no size score) produced most diverse Phase 1 data but tested on 15 features only — needs reconfirmation on 30 features
- Sampled 50k dataset produced highest absolute scores but proxy added no value — problem too easy
- RandomForest + UCB did not beat Ridge alone in any configuration tested

## Phase 1 Heuristic Ablation — Round 1 (30 features)

| Heuristic                   | Score Range | Layer Configs | Act Combos | Max Depth | Diversity Score |
|-----------------------------|-------------|---------------|------------|-----------|-----------------|
| Naive (baseline)            | 0.224       | 8             | 12         | 2         | 0.360           |
| A — equal weights           | 0.199       | 10            | 23         | 3         | 0.417           |
| B — no depth                | 0.181       | 11            | 15         | 3         | 0.359           |
| C — no activation diversity | 0.191       | 10            | 19         | 3         | 0.385           |
| D — no exploration decay    | 0.173       | 10            | 18         | 3         | 0.361           |
| E — no size score           | 0.197       | 10            | 21         | 3         | 0.403           |
| Diversity                   | 0.190       | 10            | 16         | 3         | 0.366           |

## Key Findings (Round 1, 30 features)

- Heuristic A (equal weights) won on 30 features with diversity score 0.417
- Heuristic E (no size score) came second at 0.403 — was the winner on 15 features
- All diversity-aware heuristics explored 3-layer architectures — naive never did
- Removing depth bonus (B) consistently hurts across both 15 and 30 feature runs
- Removing exploration decay (D) produces the lowest score range — hurts proxy signal
- Naive heuristic is worst in all meaningful metrics despite highest score range
- A and E are close enough that Round 2 will explore variations around both

## Current Approach

Restarting heuristic ablation on 30 raw features to confirm Round 1 findings before proceeding to Round 2 and final benchmark run.

- Phase 2 UCB beta sweep: test beta = 0.0, 0.1, 0.3, 0.5, 1.0 with winning Phase 1 heuristic on 30 features