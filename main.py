import csv
import os
from search.astar import astar_search
from train.creditcard_trainer import evaluate_architecture
from proxy.predictor import ProxyModel

def phase1_collect_data(budget=150, heuristic='naive'):
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

def phase2_proxy_search(budget=150):
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
    # Phase 1 — diversity heuristic, budget=150
    phase1_path = 'results/phase1_diversity.csv'
    if not os.path.exists(phase1_path):
        print("\nRunning Phase 1 with diversity heuristic, budget=150...")
        results = astar_search(
            evaluate_fn=evaluate_architecture,
            budget=150,
            use_proxy=False,
            results_path=phase1_path,
            heuristic='diversity'
        )
        print(f"Phase 1 complete. {len(results)} architectures saved.")
    else:
        print("Phase 1 already exists, skipping.")

    # Phase 2 — RF proxy, beta=0.0, budget=150
    phase2_path = 'results/phase2_diversity_rf.csv'
    if not os.path.exists(phase2_path):
        print("\nRunning Phase 2 with diversity heuristic, RF proxy, beta=0.0...")
        proxy = ProxyModel(model_type='rf')
        proxy.train(phase1_path)
        results = astar_search(
            evaluate_fn=evaluate_architecture,
            budget=150,
            use_proxy=True,
            proxy=proxy,
            results_path=phase2_path,
            beta=0.0
        )
        results.sort(key=lambda x: x['val_acc'], reverse=True)
        print(f"Phase 2 best: {results[0]['val_acc']:.4f} {results[0]['architecture'].hidden_layers}")
    else:
        print("Phase 2 already exists, skipping.")