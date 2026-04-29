def naive_heuristic(arch):
    from search.space import INPUT_SIZE, OUTPUT_SIZE
    max_params = INPUT_SIZE * 512 + 512 * OUTPUT_SIZE
    return 1.0 - (arch.param_count() / max_params)

def _size_score(arch):
    from search.space import INPUT_SIZE, OUTPUT_SIZE
    max_params = INPUT_SIZE * 512 + 512 * OUTPUT_SIZE
    return 1.0 - (arch.param_count() / max_params)

def _depth_score(arch):
    return len(arch.hidden_layers) / 5

def _exploration_bonus(t, budget):
    return 1.0 - (t / budget)

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

# round 1 heuristics — component ablation
def diversity_heuristic(arch, visited, t, budget):
    return (
        0.5 * _size_score(arch) +
        0.1 * _depth_score(arch) +
        0.2 * _activation_diversity_score(arch, visited) +
        0.2 * _exploration_bonus(t, budget)
    )

def heuristic_A(arch, visited, t, budget):
    # equal weights
    return (
        0.25 * _size_score(arch) +
        0.25 * _depth_score(arch) +
        0.25 * _activation_diversity_score(arch, visited) +
        0.25 * _exploration_bonus(t, budget)
    )

def heuristic_B(arch, visited, t, budget):
    # no depth bonus
    return (
        0.50 * _size_score(arch) +
        0.25 * _activation_diversity_score(arch, visited) +
        0.25 * _exploration_bonus(t, budget)
    )

def heuristic_C(arch, visited, t, budget):
    # no activation diversity
    return (
        0.50 * _size_score(arch) +
        0.25 * _depth_score(arch) +
        0.25 * _exploration_bonus(t, budget)
    )

def heuristic_D(arch, visited, t, budget):
    # no exploration decay
    return (
        0.50 * _size_score(arch) +
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
    'naive':      lambda arch, visited, t, budget: naive_heuristic(arch),
    'diversity':  diversity_heuristic,
    'A':          heuristic_A,
    'B':          heuristic_B,
    'C':          heuristic_C,
    'D':          heuristic_D,
    'E':          heuristic_E,
}
