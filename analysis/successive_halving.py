import csv
import random
import numpy as np
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

def successive_halving(
    budget=50,
    min_epochs=2,
    max_epochs=10,
    eta=2,
    results_path='results/successive_halving.csv',
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)

    start_time = datetime.now()
    print(f"[{start_time.strftime('%H:%M:%S')}] SuccessiveHalving started")
    print(f"  budget={budget} | min_epochs={min_epochs} | max_epochs={max_epochs} | eta={eta}")

    # compute number of rounds and initial candidates
    import math
    n_rounds = math.ceil(math.log(max_epochs / min_epochs, eta)) + 1
    n_candidates = budget

    print(f"  rounds={n_rounds} | initial candidates={n_candidates}")

    # sample initial candidates
    candidates = []
    seen = set()
    while len(candidates) < n_candidates:
        arch = random_architecture()
        if arch not in seen:
            seen.add(arch)
            candidates.append(arch)

    all_results = []
    first_write = True
    fieldnames = ['round', 'epochs', 'layers', 'activations', 'dropout_rates',
                  'learning_rate', 'val_score', 'train_time', 'param_count']

    for r in range(n_rounds):
        epochs = min(min_epochs * (eta ** r), max_epochs)
        epochs = int(epochs)
        n_keep = max(1, len(candidates) // eta)

        round_time = datetime.now()
        print(f"\n[{round_time.strftime('%H:%M:%S')}] Round {r+1}/{n_rounds} | "
              f"candidates={len(candidates)} | epochs={epochs} | keeping top {n_keep}")

        round_results = []
        for i, arch in enumerate(candidates):
            val_acc, train_time, params = evaluate_architecture(arch, epochs=epochs)
            round_results.append({
                'arch': arch,
                'val_acc': val_acc,
                'train_time': train_time,
                'params': params
            })

            now = datetime.now().strftime('%H:%M:%S')
            print(f"  [{now}] {i+1}/{len(candidates)} | layers={arch.hidden_layers} | "
                  f"val_acc={val_acc:.4f} | epochs={epochs}")

            with open(results_path, 'w' if first_write else 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if first_write:
                    writer.writeheader()
                    first_write = False
                writer.writerow({
                    'round': r + 1,
                    'epochs': epochs,
                    'layers': arch.hidden_layers,
                    'activations': arch.activations,
                    'dropout_rates': arch.dropout_rates,
                    'learning_rate': arch.learning_rate,
                    'val_score': round(val_acc, 4),
                    'train_time': round(train_time, 2),
                    'param_count': params
                })

        round_results.sort(key=lambda x: x['val_acc'], reverse=True)
        all_results.extend(round_results)

        print(f"  Round {r+1} best: {round_results[0]['val_acc']:.4f} "
              f"{round_results[0]['arch'].hidden_layers}")

        if r < n_rounds - 1:
            candidates = [r['arch'] for r in round_results[:n_keep]]

    all_results.sort(key=lambda x: x['val_acc'], reverse=True)
    best = all_results[0]

    end_time = datetime.now()
    elapsed = (end_time - start_time).seconds // 60
    print(f"\n[{end_time.strftime('%H:%M:%S')}] SuccessiveHalving complete | total={elapsed}m")
    print(f"Best: {best['val_acc']:.4f} {best['arch'].hidden_layers} "
          f"(params={best['params']})")

    return all_results

if __name__ == '__main__':
    successive_halving(
        budget=150,
        min_epochs=2,
        max_epochs=10,
        eta=2
    )