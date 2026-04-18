from search.space import Architecture
from search.operators import add_layer, remove_layer, widen_layer, narrow_layer, get_successors

def test_add_layer_increases_depth():
    arch = Architecture(hidden_layers=[64])
    new_arch = add_layer(arch)
    assert len(new_arch.hidden_layers) == 2

def test_remove_layer_decreases_depth():
    arch = Architecture(
        hidden_layers=[128, 64],
        activations=['relu', 'relu'],
        dropout_rates=[0.0, 0.0]
    )
    new_arch = remove_layer(arch)
    assert new_arch is not None
    assert len(new_arch.hidden_layers) == 1

def test_remove_layer_fails_on_single_layer():
    arch = Architecture(hidden_layers=[64])
    assert remove_layer(arch) is None

def test_widen_layer_increases_size():
    arch = Architecture(hidden_layers=[64])
    new_arch = widen_layer(arch, 0)
    assert new_arch.hidden_layers[0] == 128

def test_narrow_layer_decreases_size():
    arch = Architecture(hidden_layers=[64])
    new_arch = narrow_layer(arch, 0)
    assert new_arch.hidden_layers[0] == 32

def test_get_successors_returns_valid_architectures():
    arch = Architecture(hidden_layers=[64])
    successors = get_successors(arch)
    assert len(successors) > 0
    assert all(s.is_valid() for s in successors)

def test_original_arch_unchanged_after_mutation():
    arch = Architecture(hidden_layers=[64])
    add_layer(arch)
    assert len(arch.hidden_layers) == 1