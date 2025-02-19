import pytest
import torch
from non_uniform_piecewise_layers.adaptive_piecewise_mingru import (
    prefix_sum_hidden_states,
    MinGRULayer,
    MinGRUStack,
    solve_recurrence,
    #solve_scalar_recurrence
)

@pytest.fixture
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

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
    x = torch.randn(1, T, input_dim)  # Add batch dimension
    h = torch.zeros(1, state_dim)  # Add batch dimension
    
    ht = layer(x, h)
    
    # Hidden states should be 3D (B, T, state_dim)
    assert ht.shape == (1, T, state_dim)
    
    # Test batched forward pass
    B = 3
    x_batch = torch.randn(B, T, input_dim)
    h_batch = torch.zeros(B, state_dim)
    
    ht_batch = layer(x_batch, h_batch)
    
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
    T = 10
    x = torch.ones(1, T, input_dim) * 1e6  # Add batch dimension
    h = torch.zeros(1, state_dim)  # Add batch dimension
    
    ht = layer(x, h)
    
    assert not torch.isnan(ht).any()
    assert ht.shape == (1, T, state_dim)
    
    # Test with very small inputs
    x = torch.ones(1, T, input_dim) * 1e-6  # Add batch dimension
    ht = layer(x, h)
    
    assert not torch.isnan(ht).any()
    assert ht.shape == (1, T, state_dim)

def test_solve_recurrence():
    # Test dimensions
    B, T, V = 2, 4, 3
    
    # Create test inputs with controlled values for easier debugging
    a = torch.ones(B, T, V) * 0.5  # All coefficients are 0.5
    b = torch.ones(B, T, V)        # All additive terms are 1
    h0 = torch.zeros(B, V)         # Initial conditions are 0
    
    # Get the solution from solve_recurrence
    h = solve_recurrence(a, b, h0)
    
    # Print intermediate values for debugging
    print("\nTest values:")
    print(f"a shape: {a.shape}, values:\n{a[0, :, 0]}")  # First batch, first feature
    print(f"b shape: {b.shape}, values:\n{b[0, :, 0]}")  # First batch, first feature
    print(f"h0 shape: {h0.shape}, values:\n{h0[0, :]}")  # First batch
    print(f"h shape: {h.shape}, values:\n{h[0, :, 0]}")  # First batch, first feature
    
    # Verify shape
    assert h.shape == (B, T, V)
    
    # Verify the recurrence relation at each time step
    h_prev = h0
    for t in range(T):
        h_t = a[:, t, :] * h_prev + b[:, t, :]
        print(f"\nTime step {t}:")
        print(f"h_prev: {h_prev[0, 0]}")
        print(f"Expected h_t: {h_t[0, 0]}")
        print(f"Actual h_t: {h[0, t, 0]}")
        torch.testing.assert_close(h[:, t, :], h_t, rtol=1e-4, atol=1e-4)
        h_prev = h[:, t, :]

def test_solve_recurrence_random():
    # Test dimensions
    B, T, V = 2, 4, 3
    
    # Create test inputs with random values
    torch.manual_seed(42)  # For reproducibility
    a = torch.rand(B, T, V)  # Random coefficients between 0 and 1
    b = torch.randn(B, T, V)  # Random additive terms from normal distribution
    h0 = torch.zeros(B, V)  # Initialize with zeros for simpler verification
    
    # Get the solution from solve_recurrence
    h = solve_recurrence(a, b, h0)
    
    # Verify shape
    assert h.shape == (B, T, V)
    
    # Verify the recurrence relation at each time step
    h_prev = h0
    for t in range(T):
        h_t = a[:, t, :] * h_prev + b[:, t, :]
        torch.testing.assert_close(h[:, t, :], h_t, rtol=1e-4, atol=1e-4)
        h_prev = h[:, t, :]  # Use the actual computed values for next iteration
