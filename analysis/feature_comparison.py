import json
import numpy as np
from search.space import Architecture
from train.creditcard_trainer import evaluate_architecture, set_selected_features

def run_comparison(n_runs=3):
    best_arch = Architecture(
        hidden_layers=[128],
        activations=['relu'],
        dropout_rates=[0.0],
        learning_rate=0.001
    )

    with open('results/forward_selection.json', 'r') as f:
        fs = json.load(f)
    selected_indices = fs['indices']

    print("=" * 50)
    print("Post-NAS Feature Comparison")
    print("Architecture: [128] relu")
    print(f"Runs: {n_runs}")
    print("=" * 50)

    # 30 features
    print("\n30 features (all):")
    set_selected_features(None)
    scores_30 = []
    for i in range(n_runs):
        auc_pr, train_time, params = evaluate_architecture(best_arch)
        scores_30.append(auc_pr)
        print(f"  Run {i+1}: {auc_pr:.4f} ({train_time:.1f}s)")
    mean_30 = np.mean(scores_30)
    std_30 = np.std(scores_30)
    print(f"  Mean: {mean_30:.4f} +/- {std_30:.4f}")

    # 15 features
    print("\n15 features (forward selection):")
    set_selected_features(selected_indices)
    scores_15 = []
    for i in range(n_runs):
        auc_pr, train_time, params = evaluate_architecture(best_arch)
        scores_15.append(auc_pr)
        print(f"  Run {i+1}: {auc_pr:.4f} ({train_time:.1f}s)")
    mean_15 = np.mean(scores_15)
    std_15 = np.std(scores_15)
    print(f"  Mean: {mean_15:.4f} +/- {std_15:.4f}")

    # reset
    set_selected_features(None)

    print("\n" + "=" * 50)
    print(f"30 features: {mean_30:.4f} +/- {std_30:.4f}")
    print(f"15 features: {mean_15:.4f} +/- {std_15:.4f}")
    print(f"Difference:  {mean_30 - mean_15:+.4f} ({'30 features wins' if mean_30 > mean_15 else '15 features wins'})")
    print("=" * 50)

if __name__ == '__main__':
    run_comparison(n_runs=3)