import pandas as pd
import ast
import sys
import os

def analyze_phase1(csv_path='results/phase1_naive.csv'):
    df = pd.read_csv(csv_path)
    df['layers'] = df['layers'].apply(ast.literal_eval)
    df['activations'] = df['activations'].apply(ast.literal_eval)

    print(f"\nAnalyzing: {csv_path}")
    print("=" * 50)

    # 1. score range
    score_min = df['val_score'].min()
    score_max = df['val_score'].max()
    score_range = score_max - score_min
    print(f"Score range:       {score_min:.4f} - {score_max:.4f} (range: {score_range:.4f})")

    # 2. distinct layer configurations
    layer_configs = df['layers'].apply(str).nunique()
    print(f"Distinct layer configs:     {layer_configs}")

    # 3. distinct activation combinations
    activation_combos = df['activations'].apply(str).nunique()
    print(f"Distinct activation combos: {activation_combos}")

    # 4. layer depth distribution
    df['depth'] = df['layers'].apply(len)
    depth_counts = df['depth'].value_counts().sort_index()
    print(f"\nDepth distribution:")
    for depth, count in depth_counts.items():
        print(f"  {depth} layer(s): {count} evaluations")

    # 5. layer size coverage
    all_sizes = [size for layers in df['layers'] for size in layers]
    size_counts = pd.Series(all_sizes).value_counts().sort_index()
    print(f"\nLayer size coverage:")
    for size, count in size_counts.items():
        print(f"  {size}: {count} occurrences")

    # 6. activation coverage
    all_acts = [act for acts in df['activations'] for act in acts]
    act_counts = pd.Series(all_acts).value_counts()
    print(f"\nActivation coverage:")
    for act, count in act_counts.items():
        print(f"  {act}: {count} occurrences")

    # note: empirical Phase 2 validation showed diversity score does not
    # reliably predict Phase 2 AUC-PR — use as a structural diagnostic
    # only, not as a decision metric for heuristic selection

    print("=" * 50)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        analyze_phase1(sys.argv[1])
    else:
        for h in ['naive', 'A', 'B', 'C', 'D', 'E', 'diversity']:
            path = f'results/phase1_{h}.csv'
            if os.path.exists(path):
                analyze_phase1(path)
            else:
                print(f"\nSkipping {path} — file not found")