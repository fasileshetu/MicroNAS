from search.space import Architecture

def test_default_architecture_is_valid():
    arch = Architecture()
    assert arch.is_valid()

def test_param_count_is_correct():
    arch = Architecture(hidden_layers=[128])
    # INPUT_SIZE * 128 + 128 * OUTPUT_SIZE
    from search.space import INPUT_SIZE, OUTPUT_SIZE
    assert arch.param_count() == INPUT_SIZE * 128 + 128 * OUTPUT_SIZE

def test_empty_layers_is_invalid():
    arch = Architecture(hidden_layers=[], activations=[], dropout_rates=[])
    assert not arch.is_valid()

def test_too_many_layers_is_invalid():
    arch = Architecture(
        hidden_layers=[32, 32, 32, 32, 32, 32],
        activations=['relu'] * 6,
        dropout_rates=[0.0] * 6
    )
    assert not arch.is_valid()

def test_mismatched_lists_is_invalid():
    arch = Architecture(
        hidden_layers=[32, 64],
        activations=['relu'],
        dropout_rates=[0.0]
    )
    assert not arch.is_valid()

def test_architecture_is_hashable():
    arch1 = Architecture(hidden_layers=[64], activations=['relu'], dropout_rates=[0.0])
    arch2 = Architecture(hidden_layers=[64], activations=['relu'], dropout_rates=[0.0])
    assert hash(arch1) == hash(arch2)
    assert arch1 == arch2