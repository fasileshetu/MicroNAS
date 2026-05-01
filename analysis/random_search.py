import random
import numpy as np
import csv
from datetime import datetime
from search.space import Architecture, VALID_ACTIVATIONS, VALID_LAYER_SIZES
from train.creditcard_trainer import evaluate_architecture

def random_architecture():
    depth = random.randint(1, 5)
    return Architecture(
        hidden_layers=[random.choice(VALID_LAYER_SIZES) for _ in range(depth)],
        activations=[random.choice(VALID_ACTIVATIONS) for _ in range(depth)],
        dropout_rates=[0.0] * depth,
        learning_rate=0.001
    )

def random_search(budget=150, results_path='results/random_search.csv', seed=42):
    random.seed(seed)
    np.random.seed(seed)

    start_time = datetime.now()
    print(f"[{start_time.strftime('%H:%M:%S')}] Random search started | budget={budget} | seed={seed}")

    results = []
    visited = set()
    first_write = True

    fieldnames = ['layers', 'activations', 'dropout_rates', 'learning_rate',
                  'val_score', 'train_time', 'param_count']

    while len(results) < budget:
        arch = random_architecture()
        if arch in visited:
            continue
        visited.add(arch)

        val_acc, train_time, params = evaluate_architecture(arch)
        results.append({'architecture': arch, 'val_acc': val_acc})

        with open(results_path, 'w' if first_write else 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if first_write:
                writer.writeheader()
                first_write = False
            writer.writerow({
                'layers': arch.hidden_layers,
                'activations': arch.activations,
                'dropout_rates': arch.dropout_rates,
                'learning_rate': arch.learning_rate,
                'val_score': round(val_acc, 4),
                'train_time': round(train_time, 2),
                'param_count': params
            })

        now = datetime.now().strftime('%H:%M:%S')
        print(f"[{now}] {len(results)}/{budget} | layers={arch.hidden_layers} | val_acc={val_acc:.4f} | params={params}")

    results.sort(key=lambda x: x['val_acc'], reverse=True)
    best = results[0]

    end_time = datetime.now()
    elapsed = (end_time - start_time).seconds // 60
    print(f"[{end_time.strftime('%H:%M:%S')}] Random search complete | total={elapsed}m")
    print(f"Best: {best['val_acc']:.4f} {best['architecture'].hidden_layers}")

    return results

if __name__ == '__main__':
    random_search(budget=150)