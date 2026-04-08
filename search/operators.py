from copy import deepcopy
from search.space import Architecture, VALID_ACTIVATIONS, VALID_LAYER_SIZES

def add_layer(arch: Architecture):
    if len(arch.hidden_layers) >= 5:
        return None
    new = deepcopy(arch)
    new.hidden_layers.append(128)
    new.activations.append('relu')
    new.dropout_rates.append(0.0)
    return new if new.is_valid() else None

def remove_layer(arch: Architecture):
    if len(arch.hidden_layers) <= 1:
        return None
    new = deepcopy(arch)
    new.hidden_layers.pop()
    new.activations.pop()
    new.dropout_rates.pop()
    return new if new.is_valid() else None

def widen_layer(arch: Architecture, idx: int):
    new = deepcopy(arch)
    current = new.hidden_layers[idx]
    current_pos = VALID_LAYER_SIZES.index(current)
    if current_pos >= len(VALID_LAYER_SIZES) - 1:
        return None
    new.hidden_layers[idx] = VALID_LAYER_SIZES[current_pos + 1]
    return new if new.is_valid() else None

def narrow_layer(arch: Architecture, idx: int):
    new = deepcopy(arch)
    current = new.hidden_layers[idx]
    current_pos = VALID_LAYER_SIZES.index(current)
    if current_pos <= 0:
        return None
    new.hidden_layers[idx] = VALID_LAYER_SIZES[current_pos - 1]
    return new if new.is_valid() else None

def change_activation(arch: Architecture, idx: int, new_act: str):
    if new_act == arch.activations[idx]:
        return None
    new = deepcopy(arch)
    new.activations[idx] = new_act
    return new if new.is_valid() else None

def get_successors(arch: Architecture):
    successors = []

    # add/remove layers
    for op in [add_layer, remove_layer]:
        result = op(arch)
        if result:
            successors.append(result)

    # widen/narrow each layer
    for i in range(len(arch.hidden_layers)):
        for op in [widen_layer, narrow_layer]:
            result = op(arch, i)
            if result:
                successors.append(result)

    # change activation on each layer
    for i in range(len(arch.hidden_layers)):
        for act in VALID_ACTIVATIONS:
            result = change_activation(arch, i, act)
            if result:
                successors.append(result)

    return successors