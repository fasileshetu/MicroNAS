import csv
import os
from search.astar import astar_search
from train.creditcard_trainer import evaluate_architecture
from proxy.predictor import ProxyModel

def phase1_collect_data(budget=50, heuristic='naive'):
    print("-" * 50)
    print(f"PHASE 1: Collecting training data with heuristic '{heuristic}'")
    print("-" * 50)
    results = astar_search(
        evaluate_fn=evaluate_architecture,
        budget=budget,
        use_proxy=False,
        results_path=f'results/phase1_{heuristic}.csv',
        heuristic=heuristic
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
    # running on 30 raw features — no forward selection
    for h in ['F', 'G', 'H', 'I', 'J', 'K']:
        path = f'results/phase1_{h}.csv'
        if not os.path.exists(path):
            print(f"\nRunning Phase 1 with heuristic '{h}'...")
            phase1_collect_data(budget=50, heuristic=h)
        else:
            print(f"Phase 1 heuristic '{h}' already exists, skipping.")

    # phase 2 commented out until best phase 1 heuristic is determined
    # results2 = phase2_proxy_search()
    # print_best(results2, phase=2)