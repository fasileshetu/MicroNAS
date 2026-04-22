import heapq
import csv
from search.space import Architecture
from search.operators import get_successors

def naive_heuristic(arch):
    from search.space import INPUT_SIZE, OUTPUT_SIZE
    max_params = INPUT_SIZE * 512 + 512 * OUTPUT_SIZE
    return 1.0 - (arch.param_count() / max_params)

def _activation_diversity_score(arch, visited):
    activation_counts = {'relu': 0, 'tanh': 0, 'sigmoid': 0}
    for v in visited:
        for act in v.activations:
            if act in activation_counts:
                activation_counts[act] += 1
    total = sum(activation_counts.values()) + 1
    return sum(
        1.0 - (activation_counts[act] / total)
        for act in set(arch.activations)
    ) / len(arch.activations)

def _size_score(arch):
    from search.space import INPUT_SIZE, OUTPUT_SIZE
    max_params = INPUT_SIZE * 512 + 512 * OUTPUT_SIZE
    return 1.0 - (arch.param_count() / max_params)

def _depth_score(arch):
    return len(arch.hidden_layers) / 5

def _exploration_bonus(t, budget):
    return 1.0 - (t / budget)

def diversity_v2(arch, visited, t, budget):
    return (
        0.5 * _size_score(arch) +
        0.1 * _depth_score(arch) +
        0.2 * _activation_diversity_score(arch, visited) +
        0.2 * _exploration_bonus(t, budget)
    )

def heuristic_A(arch, visited, t, budget):
    # equal weights baseline
    return (
        0.25 * _size_score(arch) +
        0.25 * _depth_score(arch) +
        0.25 * _activation_diversity_score(arch, visited) +
        0.25 * _exploration_bonus(t, budget)
    )

def heuristic_B(arch, visited, t, budget):
    # no depth bonus
    return (
        0.5 * _size_score(arch) +
        0.25 * _activation_diversity_score(arch, visited) +
        0.25 * _exploration_bonus(t, budget)
    )

def heuristic_C(arch, visited, t, budget):
    # no activation diversity
    return (
        0.5 * _size_score(arch) +
        0.25 * _depth_score(arch) +
        0.25 * _exploration_bonus(t, budget)
    )

def heuristic_D(arch, visited, t, budget):
    # no exploration decay
    return (
        0.5 * _size_score(arch) +
        0.25 * _depth_score(arch) +
        0.25 * _activation_diversity_score(arch, visited)
    )

def heuristic_E(arch, visited, t, budget):
    # no size score
    return (
        0.33 * _depth_score(arch) +
        0.33 * _activation_diversity_score(arch, visited) +
        0.33 * _exploration_bonus(t, budget)
    )

HEURISTICS = {
    'naive': lambda arch, visited, t, budget: naive_heuristic(arch),
    'diversity_v2': diversity_v2,
    'A': heuristic_A,
    'B': heuristic_B,
    'C': heuristic_C,
    'D': heuristic_D,
    'E': heuristic_E,
}

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
