import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

def load_features(path='data/creditcard.csv'):
    df = pd.read_csv(path)
    scaler = StandardScaler()
    df['Time'] = scaler.fit_transform(df[['Time']])
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    X = df.drop('Class', axis=1).values.astype('float32')
    y = df['Class'].values.astype('float32')
    return X, y, list(df.drop('Class', axis=1).columns)

def evaluate_feature_subset(X, y, feature_indices):
    """
    Trains a logistic regression on the given feature subset
    and returns average AUC-PR using cross validation.
    """
    X_subset = X[:, feature_indices]
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs'
    )
    # use average precision score via cross validation
    scores = cross_val_score(
        model, X_subset, y,
        cv=3,
        scoring='average_precision'
    )
    return scores.mean()

def forward_selection(max_features=15, path='data/creditcard.csv'):
    """
    Runs forward selection on the credit card dataset.
    Starts with no features and greedily adds the most predictive one
    at each step until max_features is reached or no improvement.
    """
    X, y, feature_names = load_features(path)
    n_features = X.shape[1]

    selected = []
    remaining = list(range(n_features))
    best_score = 0.0

    print(f"Running Forward Selection on {n_features} features...")
    print("-" * 50)

    for step in range(max_features):
        step_best_score = 0.0
        step_best_feature = None

        for feature_idx in remaining:
            candidate = selected + [feature_idx]
            score = evaluate_feature_subset(X, y, candidate)

            if score > step_best_score:
                step_best_score = score
                step_best_feature = feature_idx

        # stop if adding features no longer helps
        if step_best_score <= best_score:
            print(f"No improvement at step {step + 1}, stopping early.")
            break

        selected.append(step_best_feature)
        remaining.remove(step_best_feature)
        best_score = step_best_score

        print(f"Step {step + 1}: added '{feature_names[step_best_feature]}' "
              f"| selected={[feature_names[i] for i in selected]} "
              f"| AUC-PR={best_score:.4f}")

    print("-" * 50)
    print(f"Forward Selection complete.")
    print(f"Selected {len(selected)} features: {[feature_names[i] for i in selected]}")
    print(f"Final AUC-PR: {best_score:.4f}")

    return selected, [feature_names[i] for i in selected], best_score

if __name__ == '__main__':
    import json

    selected_indices, selected_names, fs_score = forward_selection()

    with open('results/forward_selection.json', 'w') as f:
        json.dump({
            'indices': selected_indices,
            'names': selected_names,
            'score': fs_score
        }, f)

    print(f"\nSaved to results/forward_selection.json")