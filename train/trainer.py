import time
import numpy as np
from tensorflow import keras
from search.space import Architecture

def build_model(arch: Architecture) -> keras.Model:
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(784,)))
    
    for i, (size, activation, dropout) in enumerate(zip(
        arch.hidden_layers, arch.activations, arch.dropout_rates)):
        
        model.add(keras.layers.Dense(size, activation=activation))
        if dropout > 0:
            model.add(keras.layers.Dropout(dropout))
    
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

def load_mnist_subset(num_train=5000, num_val=1000):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    # use a small subset for speed
    x_train, y_train = x_train[:num_train], y_train[:num_train]
    x_val, y_val = x_test[:num_val], y_test[:num_val]
    
    return (x_train, y_train), (x_val, y_val)

def evaluate_architecture(arch: Architecture) -> tuple:
    """
    Trains arch on MNIST subset and returns (val_accuracy, train_time, param_count).
    This is the evaluate_fn passed into astar_search.
    """
    (x_train, y_train), (x_val, y_val) = load_mnist_subset()
    
    model = build_model(arch)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=arch.learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    start = time.time()
    model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=64,
        verbose=0
    )
    train_time = time.time() - start
    
    _, val_acc = model.evaluate(x_val, y_val, verbose=0)
    param_count = arch.param_count()
    
    return (val_acc, train_time, param_count)