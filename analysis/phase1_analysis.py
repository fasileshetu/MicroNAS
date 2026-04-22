import pandas as pd
import ast
import sys

def analyze_phase1(csv_path='results/creditcard_phase1.csv'):
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

    # 7. summary score
    # higher is better — rewards diversity and score range
    diversity_score = (layer_configs / 50) * 0.4 + \
                      (activation_combos / 50) * 0.3 + \
                      (score_range / 0.3) * 0.3
    print(f"\nDiversity score (higher = better proxy training data): {diversity_score:.4f}")
    print("=" * 50)

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'results/creditcard_phase1.csv'
    analyze_phase1(path)