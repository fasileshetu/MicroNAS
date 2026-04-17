import time
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from search.space import Architecture

SELECTED_FEATURES = None

def set_selected_features(indices):
    global SELECTED_FEATURES
    SELECTED_FEATURES = indices

def build_model(arch: Architecture, input_size: int) -> keras.Model:
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_size,)))

    for size, activation, dropout in zip(
        arch.hidden_layers, arch.activations, arch.dropout_rates
    ):
        model.add(keras.layers.Dense(size, activation=activation))
        if dropout > 0:
            model.add(keras.layers.Dropout(dropout))

    model.add(keras.layers.Dense(1, activation='sigmoid'))
    return model

def load_creditcard(path='data/creditcard.csv'):
    df = pd.read_csv(path)

    scaler = StandardScaler()
    df['Time'] = scaler.fit_transform(df[['Time']])
    df['Amount'] = scaler.fit_transform(df[['Amount']])

    X = df.drop('Class', axis=1).values.astype('float32')
    y = df['Class'].values.astype('float32')

    if SELECTED_FEATURES is not None:
        X = X[:, SELECTED_FEATURES]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return (X_train, y_train), (X_val, y_val)

def evaluate_architecture(arch: Architecture) -> tuple:
    (x_train, y_train), (x_val, y_val) = load_creditcard()

    fraud_count = y_train.sum()
    legit_count = len(y_train) - fraud_count
    class_weight = {0: 1.0, 1: legit_count / fraud_count}

    input_size = x_train.shape[1]
    model = build_model(arch, input_size=input_size)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=arch.learning_rate, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=3,
        restore_best_weights=True
    )

    start = time.time()
    model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=64,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=0
    )
    train_time = time.time() - start

    y_pred = model.predict(x_val, verbose=0).flatten()
    auc_pr = average_precision_score(y_val, y_pred)

    return (auc_pr, train_time, arch.param_count())