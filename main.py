import csv
import os
from search.astar import astar_search
from train.trainer import evaluate_architecture
from proxy.predictor import ProxyModel

def save_results(results, path='results/log_phase1.csv'):
    fieldnames = ['layers', 'activations', 'dropout_rates', 'learning_rate',
                  'val_acc', 'train_time', 'param_count']

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            arch = r['architecture']
            writer.writerow({
                'layers': arch.hidden_layers,
                'activations': arch.activations,
                'dropout_rates': arch.dropout_rates,
                'learning_rate': arch.learning_rate,
                'val_acc': round(r['val_acc'], 4),
                'train_time': round(r['train_time'], 2),
                'param_count': r['param_count']
            })

def phase1_collect_data(budget=50):
    print("-" * 50)
    print("PHASE 1: Collecting training data with epsilon greedy heuristic")
    print("-" * 50)
    results = astar_search(
        evaluate_fn=evaluate_architecture,
        budget=budget,
        use_proxy=False
    )
    save_results(results)
    print(f"\nPhase 1 complete. {len(results)} architectures saved to results/log_phase1.csv")
    return results

def phase2_proxy_search(budget=100):
    print("-" * 50)
    print("PHASE 2: Training proxy model and running guided search")
    print("-" * 50)
    proxy = ProxyModel()
    proxy.train('results/log_phase1.csv')

    results = astar_search(
        evaluate_fn=evaluate_architecture,
        budget=budget,
        use_proxy=True,
        proxy=proxy
    )
    save_results(results, path='results/log_phase2.csv')
    print(f"\nPhase 2 complete. Results saved to results/log_phase2.csv")
    return results

def print_best(results, phase):
    best = results[0]
    print(f"\nBest architecture found in Phase {phase}:")
    print(f"  Layers: {best['architecture'].hidden_layers}")
    print(f"  Activations: {best['architecture'].activations}")
    print(f"  Val accuracy: {best['val_acc']:.4f}")
    print(f"  Param count: {best['param_count']}")

if __name__ == '__main__':
    # Phase 1 — only run if we don't have data yet
    if not os.path.exists('results/log.csv'):
        results1 = phase1_collect_data(budget=50)
    else:
        print("Phase 1 data already exists, skipping to Phase 2")

    # Phase 2 — proxy guided search
    results2 = phase2_proxy_search(budget=50)

    print_best(results2, phase=2)