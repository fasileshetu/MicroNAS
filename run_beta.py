import sys
from datetime import datetime
from search.astar import astar_search
from train.creditcard_trainer import evaluate_architecture
from proxy.predictor import ProxyModel

def run_beta(beta):
    path = f'results/phase2_diversity_rf_ucb_{beta}.csv'
    start_time = datetime.now()
    print(f"[{start_time.strftime('%H:%M:%S')}] Starting beta={beta}...")

    proxy = ProxyModel(model_type='rf')
    proxy.train('results/phase1_diversity.csv')
    proxy_time = datetime.now()
    print(f"[{proxy_time.strftime('%H:%M:%S')}] Proxy trained for beta={beta} ({(proxy_time - start_time).seconds}s)")

    results = astar_search(
        evaluate_fn=evaluate_architecture,
        budget=150,
        use_proxy=True,
        proxy=proxy,
        results_path=path,
        beta=beta
    )
    results.sort(key=lambda x: x['val_acc'], reverse=True)

    end_time = datetime.now()
    elapsed = (end_time - start_time).seconds // 60
    print(f"[{end_time.strftime('%H:%M:%S')}] Done beta={beta} in {elapsed}m | best={results[0]['val_acc']:.4f} {results[0]['architecture'].hidden_layers}")

if __name__ == '__main__':
    beta = float(sys.argv[1])
    run_beta(beta)