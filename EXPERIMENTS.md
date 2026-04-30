# Experiments

All experiments run on the Credit Card Fraud Detection dataset (ULB).
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Metric: AUC-PR (area under precision-recall curve).
Budget: 50 evaluations per phase, 10 epochs per evaluation.

## Results

| Run                             | Features | Heuristic | Proxy    | Phase 1 | Phase 2 | Gap    |
|---------------------------------|----------|-----------|----------|---------|---------|--------|
| MNIST baseline                  | 784      | Naive     | Ridge    | 91.8%   | 92.6%   | +0.8%  |
| Sampled 50k, naive + Ridge      | 30       | Naive     | Ridge    | 0.9188  | 0.9188  | +0.000 |
| naive + Ridge                   | 30       | Naive     | Ridge    | 0.7978  | 0.8145  | +0.017 |
| naive + Ridge                   | 15       | Naive     | Ridge    | 0.8273  | 0.8105  | -0.017 |
| naive + RF                      | 15       | Naive     | RF       | 0.8015  | 0.8174  | +0.016 |
| diversity + Ridge               | 15       | Diversity | Ridge    | 0.8009  | 0.8318  | +0.032 |
| diversity + RF + UCB (beta=0.5) | 15       | Diversity | RF + UCB | 0.7997  | 0.8279  | +0.028 |
| diversity + RF + UCB (beta=1.5) | 15       | Diversity | RF + UCB | 0.7997  | 0.7969  | -0.003 |

## Phase 1 Heuristic Ablation (15 features, Round 1)

Note: diversity score was used during this phase as a proxy for training data
quality but was later abandoned after empirical Phase 2 validation showed no
correlation with Phase 2 AUC-PR. Structural metrics below are still meaningful
as diagnostics.

| Heuristic                   | Score Range | Layer Configs | Act Combos | Max Depth |
|-----------------------------|-------------|---------------|------------|-----------|
| Naive (baseline)            | 0.213       | 9             | 12         | 2         |
| A — equal weights           | 0.226       | 9             | 12         | 2         |
| B — no depth                | 0.151       | 9             | 12         | 2         |
| C — no activation diversity | 0.217       | 9             | 12         | 2         |
| D — no exploration decay    | 0.188       | 9             | 12         | 2         |
| E — no size score           | 0.227       | 12            | 16         | 3         |
| Diversity                   | 0.197       | 9             | 12         | 2         |

## Phase 1 Heuristic Ablation — Round 1 (30 features)

| Heuristic                   | Score Range | Layer Configs | Act Combos | Max Depth |
|-----------------------------|-------------|---------------|------------|-----------|
| Naive (baseline)            | 0.224       | 8             | 12         | 2         |
| A — equal weights           | 0.199       | 10            | 23         | 3         |
| B — no depth                | 0.181       | 11            | 15         | 3         |
| C — no activation diversity | 0.191       | 10            | 19         | 3         |
| D — no exploration decay    | 0.173       | 10            | 18         | 3         |
| E — no size score           | 0.197       | 10            | 21         | 3         |
| Diversity                   | 0.190       | 10            | 16         | 3         |

## Phase 2 Validation — Ridge Proxy (30 features, all 7 heuristics)

| Heuristic | Phase 2 Best | Best Architecture  |
|-----------|--------------|--------------------|
| Naive     | 0.8568       | [64, 32] tanh/relu |
| A         | 0.8504       | [128] relu         |
| D         | 0.8495       | [64] relu          |
| E         | 0.8468       | [128] relu         |
| Diversity | 0.8451       | [128] relu         |
| B         | 0.8421       | [64, 32] relu/relu |
| C         | 0.8407       | [64] relu          |

## Phase 2 Validation — RandomForest Proxy (30 features, all 7 heuristics)

| Heuristic | Phase 2 Best | Best Architecture  |
|-----------|--------------|--------------------|
| Diversity | 0.8608       | [128] relu         |
| B         | 0.8571       | [64, 32] relu/relu |
| C         | 0.8559       | [128] relu         |
| Naive     | 0.8516       | [128] relu         |
| D         | 0.8437       | [64] relu          |
| E         | 0.8416       | [128] relu         |
| A         | 0.8406       | [128] relu         |

## UCB Beta Sweep — Diversity Heuristic + RandomForest Proxy (30 features)

| Beta | Phase 2 Best | Best Architecture |
|------|--------------|-------------------|
| 0.0  | 0.8494       | [128] relu        |
| 0.1  | 0.8403       | [128] relu        |
| 0.3  | 0.8250       | [128] relu        |
| 0.5  | 0.8218       | [64] tanh         |
| 1.0  | 0.8488       | [64] relu         |

Note: beta=0.0 wins. UCB exploration adds no value for this configuration —
the proxy alone is sufficient. Results are lower than the previous best of
0.8608, consistent with run-to-run variance.

## Key Findings

- Diversity heuristic outperformed naive on 15 features (+0.032 gap vs +0.017)
- 15 features hurt proxy quality vs 30 features — cleaner data produces less score
  variance, giving the proxy less signal to learn from
- Forward selection results preserved for post-NAS experiment comparing final
  architecture performance on 30 vs 15 features
- Sampled 50k dataset produced highest absolute scores but proxy added no value —
  problem too easy on smaller dataset
- All diversity-aware heuristics on 30 features explored 3-layer architectures —
  naive never did
- Removing depth bonus (B) consistently hurts across both feature set runs
- Removing exploration decay (D) produces the lowest score range — hurts proxy signal
- Diversity score was abandoned as a decision metric — empirical Phase 2 validation
  showed no correlation with Phase 2 AUC-PR
- All heuristics produced Ridge Phase 2 results within a narrow range (0.8407-0.8568)
  — Phase 1 heuristic choice has limited impact when using Ridge proxy
- RandomForest does not consistently outperform Ridge — wins for some heuristics,
  loses for others
- Diversity heuristic + RandomForest produced the highest Phase 2 result at 0.8608
- Proxy choice and Phase 1 heuristic interact — the best proxy depends on what
  training data was collected in Phase 1
- UCB exploration adds no value — beta=0.0 wins the sweep, more exploration hurts

## Final Configuration

Heuristic: Diversity
Proxy: RandomForest
Beta: 0.0 (no UCB)
Best result: 0.8608 AUC-PR — [128] relu

## Proxy Quality

Evaluated on diversity heuristic Phase 1 training data vs Phase 2 actual results.
Kendall's Tau:  0.4008 (p=0.0000)
Top-10 Overlap: 70.00% (7/10 architectures)

## Planned Experiments

- Post-NAS feature comparison: retrain best Phase 2 architecture ([128] relu) on
  30 vs 15 features (3 runs each, averaged) to determine if forward selection
  improves final model quality independent of the search process