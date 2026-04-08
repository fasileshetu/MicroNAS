import heapq
from search.space import Architecture
from search.operators import get_successors

def astar_search(evaluate_fn, budget=50):

    start = Architecture()
    
    # f_score, counter (tiebreaker), architecture
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
        
        g = current.param_count()
        
        # h = naive heuristic for now
        h = naive_heuristic(current)
        
        for neighbor in get_successors(current):
            if neighbor not in visited:
                neighbor_g = neighbor.param_count()
                neighbor_h = naive_heuristic(neighbor)
                f_score = neighbor_g - (neighbor_h * 100000)  # scale h to matter
                counter += 1
                heapq.heappush(open_set, (f_score, counter, neighbor))
    
    # sort by val_acc descending
    results.sort(key=lambda x: x['val_acc'], reverse=True)
    return results

def naive_heuristic(arch: Architecture) -> float:
    """
    Placeholder heuristic until proxy model is trained.
    Prefers smaller networks as a rough proxy for efficiency.
    Returns a value between 0 and 1.
    """
    max_params = 784 * 512 + 512 * 10  # upper bound
    return 1.0 - (arch.param_count() / max_params)