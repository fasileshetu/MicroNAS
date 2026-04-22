import heapq
import csv
from search.space import Architecture
from search.operators import get_successors
from search.heuristics import HEURISTICS

def save_result(result, path, write_header=False):
    fieldnames = ['layers', 'activations', 'dropout_rates', 'learning_rate',
                  'val_score', 'train_time', 'param_count']
    mode = 'w' if write_header else 'a'
    with open(path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        arch = result['architecture']
        writer.writerow({
            'layers': arch.hidden_layers,
            'activations': arch.activations,
            'dropout_rates': arch.dropout_rates,
            'learning_rate': arch.learning_rate,
            'val_score': round(result['val_acc'], 4),
            'train_time': round(result['train_time'], 2),
            'param_count': result['param_count']
        })

def astar_search(evaluate_fn, budget=50, use_proxy=False, proxy=None,
                 results_path=None, heuristic='naive'):
    start = Architecture()

    counter = 0
    open_set = []
    heapq.heappush(open_set, (0.0, counter, start))

    visited = set()
    results = []
    first_write = True

    while open_set and len(results) < budget:
        f, _, current = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        val_acc, train_time, params = evaluate_fn(current)
        result = {
            'architecture': current,
            'val_acc': val_acc,
            'train_time': train_time,
            'param_count': params
        }
        results.append(result)

        if results_path:
            save_result(result, results_path, write_header=first_write)
            first_write = False

        print(f"Evaluated {len(results)}/{budget} | layers={current.hidden_layers} | val_acc={val_acc:.4f} | params={params}")

        t = len(results)

        for neighbor in get_successors(current):
            if neighbor not in visited:
                from search.space import INPUT_SIZE, OUTPUT_SIZE
                max_params = INPUT_SIZE * 512 + 512 * OUTPUT_SIZE
                g = neighbor.param_count() / max_params

                if use_proxy and proxy is not None:
                    h = proxy.predict(neighbor)
                else:
                    h = HEURISTICS[heuristic](neighbor, visited, t, budget)

                f_score = g - (h * 1)
                counter += 1
                heapq.heappush(open_set, (f_score, counter, neighbor))

    results.sort(key=lambda x: x['val_acc'], reverse=True)
    return results