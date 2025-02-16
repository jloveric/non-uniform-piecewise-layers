import pytest
import torch
from non_uniform_piecewise_layers.adaptive_piecewise_mingru import (
    prefix_sum_hidden_states,
    MinGRULayer,
    MinGRUStack
)

@pytest.fixture
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def test_prefix_sum_hidden_states_unbatched():
    # Test unbatched case
    T, D = 5, 3
    z = torch.ones(T, D) * 0.5
    h_bar = torch.ones(T, D)
    h0 = torch.zeros(D)
    
    h = prefix_sum_hidden_states(z, h_bar, h0)
    
    assert h.shape == (T, D)
    # When z=0.5, output should be mix of cumsum and h_bar
    expected = torch.ones_like(h)
    for t in range(T):
        expected[t] = 0.5 * (torch.ones(D) * (t + 1)) + 0.5 * torch.ones(D)
    
    assert torch.allclose(h, expected)

def test_prefix_sum_hidden_states_batched():
    # Test batched case
    B, T, D = 2, 4, 3
    z = torch.ones(B, T, D) * 0.5
    h_bar = torch.ones(B, T, D)
    h0 = torch.zeros(B, D)
    
    h = prefix_sum_hidden_states(z, h_bar, h0)
    
    assert h.shape == (B, T, D)

def test_mingru_layer():
    input_dim = 4
    state_dim = 4  # Match input_dim for now
    out_features = 3
    num_points = 10
    
    layer = MinGRULayer(input_dim, state_dim, out_features, num_points)
    
    # Test unbatched forward pass
    T = 6
    x = torch.randn(T, input_dim)
    h = torch.zeros(state_dim)
    
    y, ht = layer(x, h)
    
    assert y.shape == (T, out_features)
    assert ht.shape == (T, state_dim)
    
    # Test batched forward pass
    B = 3
    x_batch = torch.randn(B, T, input_dim)
    h_batch = torch.zeros(B, state_dim)
    
    y_batch, ht_batch = layer(x_batch, h_batch)
    
    assert y_batch.shape == (B, T, out_features)
    assert ht_batch.shape == (B, T, state_dim)

def test_mingru_stack():
    input_dim = 4
    state_dim = 4  # Match input_dim for now
    out_features = 3
    num_layers = 2
    num_points = 10
    
    stack = MinGRUStack(input_dim, state_dim, out_features, num_layers, num_points)
    
    # Test batched forward pass
    B, T = 3, 6
    x = torch.randn(B, T, input_dim)
    
    # Test with no initial hidden state
    output, hidden_states = stack(x)
    
    assert output.shape == (B, T, out_features)
    assert len(hidden_states) == num_layers
    assert all(h.shape == (B, T, state_dim) for h in hidden_states)
    
    # Test with provided initial hidden states
    h = [torch.zeros(B, state_dim) for _ in range(num_layers)]
    output, hidden_states = stack(x, h)
    
    assert output.shape == (B, T, out_features)
    assert len(hidden_states) == num_layers
    assert all(h.shape == (B, T, state_dim) for h in hidden_states)

def test_mingru_numerical_stability():
    # Test with extreme values
    input_dim = 4
    state_dim = 4  # Match input_dim for now
    out_features = 3
    num_points = 10
    
    layer = MinGRULayer(input_dim, state_dim, out_features, num_points)
    
    # Test with very large inputs
    x = torch.ones(10, input_dim) * 1e6
    h = torch.zeros(state_dim)
    
    y, ht = layer(x, h)
    assert not torch.isnan(y).any()
    assert not torch.isnan(ht).any()
    
    # Test with very small inputs
    x = torch.ones(10, input_dim) * 1e-6
    y, ht = layer(x, h)
    assert not torch.isnan(y).any()
    assert not torch.isnan(ht).any()
