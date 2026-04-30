import csv
import ast
import numpy as np
from scipy.stats import kendalltau
from search.space import Architecture
from proxy.predictor import ProxyModel, architecture_to_features

def load_csv(path):
    archs, scores = [], []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            arch = Architecture(
                hidden_layers=ast.literal_eval(row['layers']),
                activations=ast.literal_eval(row['activations']),
                dropout_rates=ast.literal_eval(row['dropout_rates']),
                learning_rate=float(row['learning_rate'])
            )
            archs.append(arch)
            scores.append(float(row['val_score']))
    return archs, scores

def compute_metrics(phase1_path, phase2_path, k=10):
    # train proxy on phase1 data
    proxy = ProxyModel(model_type='rf')
    proxy.train(phase1_path)

    # load phase2 results
    archs, actual_scores = load_csv(phase2_path)

    # get proxy predictions for each phase2 architecture
    predicted_scores = [proxy.predict(arch) for arch in archs]

    # kendall's tau
    tau, p_value = kendalltau(predicted_scores, actual_scores)

    # top-k overlap
    k = min(k, len(actual_scores))
    actual_top_k = set(np.argsort(actual_scores)[::-1][:k])
    predicted_top_k = set(np.argsort(predicted_scores)[::-1][:k])
    overlap = len(actual_top_k & predicted_top_k) / k

    print(f"Kendall's Tau:  {tau:.4f} (p={p_value:.4f})")
    print(f"Top-{k} Overlap: {overlap:.2%} ({len(actual_top_k & predicted_top_k)}/{k} architectures)")

    return tau, overlap

if __name__ == '__main__':
    compute_metrics(
        phase1_path='results/phase1_diversity.csv',
        phase2_path='results/phase2_diversity_rf_ucb_0.0.csv'
    )