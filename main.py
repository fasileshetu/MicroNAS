import csv
import os
import json
import search.space as space
from search.astar import astar_search
from train.creditcard_trainer import evaluate_architecture, set_selected_features
from proxy.predictor import ProxyModel

def phase1_collect_data(budget=50):
    print("-" * 50)
    print("PHASE 1: Collecting training data with naive heuristic")
    print("-" * 50)
    results = astar_search(
        evaluate_fn=evaluate_architecture,
        budget=budget,
        use_proxy=False,
        results_path='results/creditcard_phase1.csv'
    )
    print(f"\nPhase 1 complete. {len(results)} architectures saved.")
    return results

def phase2_proxy_search(budget=50):
    print("-" * 50)
    print("PHASE 2: Training proxy model and running guided search")
    print("-" * 50)
    proxy = ProxyModel()
    proxy.train('results/creditcard_phase1.csv')

    results = astar_search(
        evaluate_fn=evaluate_architecture,
        budget=budget,
        use_proxy=True,
        proxy=proxy,
        results_path='results/creditcard_phase2.csv'
    )
    print(f"\nPhase 2 complete. Results saved to results/creditcard_phase2.csv")
    return results

def print_best(results, phase):
    best = results[0]
    print(f"\nBest architecture found in Phase {phase}:")
    print(f"  Layers:     {best['architecture'].hidden_layers}")
    print(f"  Activations:{best['architecture'].activations}")
    print(f"  AUC-PR:     {best['val_acc']:.4f}")
    print(f"  Param count:{best['param_count']}")

if __name__ == '__main__':
    if not os.path.exists('results/forward_selection.json'):
        print("Forward selection results not found.")
        print("Run first: python forward_selction.py")
        exit()

    with open('results/forward_selection.json') as f:
        fs_data = json.load(f)

    selected_indices = fs_data['indices']
    selected_names = fs_data['names']
    print(f"Loaded {len(selected_indices)} selected features: {selected_names}")

    space.INPUT_SIZE = len(selected_indices)
    set_selected_features(selected_indices)

    if not os.path.exists('results/creditcard_phase1.csv'):
        results1 = phase1_collect_data()
    else:
        print("Phase 1 data already exists, skipping to Phase 2")
        with open('results/creditcard_phase1.csv') as f:
            results1 = list(csv.DictReader(f))

    results2 = phase2_proxy_search()
    print_best(results2, phase=2)