# Experiments

All experiments run on [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB).
Metric: AUC-PR (area under precision-recall curve).
Budget: 50 evaluations per phase, 10 epochs per evaluation.

## Results

| Run                        | Phase 1 | Phase 2 | Gap    |
|---------------------------|---------|---------|--------|
| Full dataset, no fixes     | 0.7978  | 0.8145  | +0.017 |
| Sampled 50k dataset        | 0.9188  | 0.9188  | +0.000 |
| Full dataset with fixes    | 0.8539  | 0.8372  | -0.017 |
| Forward selected features  | 0.8009  | 0.8277  | +0.027 |

## Key Findings

- Sampling the dataset improved absolute scores but eliminated proxy advantage
- Forward selection on 15 features gave the largest Phase 1 vs Phase 2 gap
- High variance per architecture is a known challenge with imbalanced datasets
- clipnorm=1.0 prevented catastrophic failures in deeper networks
- Correct max_params normalization was critical for proxy guided exploration

## Current Approach

Forward selection on 15 features with full 227,000 row dataset.
Phase 2 found [64, 32] sigmoid/relu at 0.8277 AUC-PR with 3,040 params
vs Phase 1 best [64] relu at 0.8009 AUC-PR with 1,024 params.