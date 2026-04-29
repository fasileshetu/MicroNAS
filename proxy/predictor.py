import numpy as np
import csv
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

VALID_ACTIVATIONS = ['relu', 'tanh', 'sigmoid']
VALID_LAYER_SIZES = [32, 64, 128, 256, 512]
MAX_LAYERS = 5

def architecture_to_features(arch) -> np.ndarray:
    features = []

    # 1. number of layers (normalized)
    features.append(len(arch.hidden_layers) / MAX_LAYERS)

    # 2. param count (normalized)
    from search.space import INPUT_SIZE, OUTPUT_SIZE
    max_params = INPUT_SIZE * 512 + 512 * OUTPUT_SIZE
    features.append(arch.param_count() / max_params)

    # 3. average layer size (normalized)
    avg_size = sum(arch.hidden_layers) / len(arch.hidden_layers)
    features.append(avg_size / max(VALID_LAYER_SIZES))

    # 4. activation counts
    for act in VALID_ACTIVATIONS:
        count = arch.activations.count(act) / MAX_LAYERS
        features.append(count)

    # 5. layer size at each position (padded to MAX_LAYERS)
    for i in range(MAX_LAYERS):
        if i < len(arch.hidden_layers):
            features.append(arch.hidden_layers[i] / max(VALID_LAYER_SIZES))
        else:
            features.append(0.0)

    # 6. activation at each position (one-hot encoded, padded to MAX_LAYERS)
    for i in range(MAX_LAYERS):
        for act in VALID_ACTIVATIONS:
            if i < len(arch.activations):
                features.append(1.0 if arch.activations[i] == act else 0.0)
            else:
                features.append(0.0)

    return np.array(features, dtype=np.float32)


class ProxyModel:
    def __init__(self, model_type='rf', n_estimators=100):
        if model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        else:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=42,
                min_samples_leaf=2
            )
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, csv_path='results/phase1_naive.csv'):
        from search.space import Architecture

        X, y = [], []

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                arch = Architecture(
                    hidden_layers=ast.literal_eval(row['layers']),
                    activations=ast.literal_eval(row['activations']),
                    dropout_rates=ast.literal_eval(row['dropout_rates']),
                    learning_rate=float(row['learning_rate'])
                )
                features = architecture_to_features(arch)
                X.append(features)
                y.append(float(row['val_score']))

        X = np.array(X)
        y = np.array(y)

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        print(f"Proxy model trained on {len(y)} architectures")
        print(f"Accuracy range in training data: {y.min():.4f} - {y.max():.4f}")

    def predict(self, arch) -> float:
        if not self.is_trained:
            raise RuntimeError("Proxy model has not been trained yet.")
        features = architecture_to_features(arch).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        return float(np.clip(prediction, 0.0, 1.0))

    def uncertainty(self, arch) -> float:
        if self.model_type != 'rf':
            raise RuntimeError("Uncertainty estimation is only available for RandomForest proxy.")
        if not self.is_trained:
            raise RuntimeError("Proxy model has not been trained yet.")
        features = architecture_to_features(arch).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        tree_predictions = np.array([
            tree.predict(features_scaled)[0]
            for tree in self.model.estimators_
        ])
        return float(np.var(tree_predictions))