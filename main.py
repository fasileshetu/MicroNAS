import os
from search.astar import astar_search
from train.creditcard_trainer import evaluate_architecture
from proxy.predictor import ProxyModel

if __name__ == '__main__':
    # Phase 1 -- diversity heuristic, budget=150
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

    # Phase 2 -- RF proxy, beta=0.0, budget=150
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