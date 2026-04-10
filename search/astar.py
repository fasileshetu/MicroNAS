import heapq
import random
from search.space import Architecture
from search.operators import get_successors

def epsilon_greedy_heuristic(arch, epsilon=0.3):
    if random.random() < epsilon:
        return random.uniform(0, 1)  # explore randomly 30% of the time
    else:
        return naive_heuristic(arch)  # exploit heuristic 70% of the time

def random_heuristic(arch):
    return random.uniform(0, 1)

def naive_heuristic(arch):
    max_params = 784 * 512 + 512 * 10
    return 1.0 - (arch.param_count() / max_params)

def astar_search(evaluate_fn, budget=50, use_proxy=False, proxy=None):
    start = Architecture()

    counter = 0
    open_set = []
    heapq.heappush(open_set, (0.0, counter, start))

    visited = set()
    results = []

    while open_set and len(results) < budget:
        f, _, current = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        val_acc, train_time, params = evaluate_fn(current)
        results.append({
            'architecture': current,
            'val_acc': val_acc,
            'train_time': train_time,
            'param_count': params
        })

        print(f"Evaluated {len(results)}/{budget} | layers={current.hidden_layers} | val_acc={val_acc:.4f} | params={params}")

        for neighbor in get_successors(current):
            if neighbor not in visited:
                max_params = 784 * 512 + 512 * 10
                g = neighbor.param_count() / max_params  # normalized 0 to 1
                if use_proxy and proxy is not None:
                    h = proxy.predict(neighbor)
                else:
                    h = naive_heuristic(neighbor)
                f_score = g - (h * 2)  # both on same scale, accuracy weighted 2x
                counter += 1
                heapq.heappush(open_set, (f_score, counter, neighbor))

    results.sort(key=lambda x: x['val_acc'], reverse=True)
    return results