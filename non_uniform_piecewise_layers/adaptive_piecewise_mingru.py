import torch
from non_uniform_piecewise_layers.adaptive_piecewise_linear import AdaptivePiecewiseLinear

def prefix_sum_hidden_states(z, h_bar, h0):
    """
    Vectorized computation of hidden states using parallel scan.
    Implements the recurrence: h[t+1] = (1-z[t])h[t] + z[t]*hbar[t]

    Args:
        z (Tensor): Update gate tensor of shape (B, T, D) or (T, D)
        h_bar (Tensor): Candidate hidden states of shape (B, T, D) or (T, D)
        h0 (Tensor): Initial hidden state of shape (B, D) or (D,)

    Returns:
        Tensor: Computed hidden states of shape (B, T, D) or (T, D)
    """
    # Add missing dimensions if needed
    if z.dim() == 1:
        z = z.unsqueeze(0).unsqueeze(0)  # Add batch and time dimensions
        h_bar = h_bar.unsqueeze(0).unsqueeze(0)
        h0 = h0.unsqueeze(0)
        unbatched = True
    elif z.dim() == 2:
        z = z.unsqueeze(0)  # Add batch dimension
        h_bar = h_bar.unsqueeze(0)
        h0 = h0.unsqueeze(0)
        unbatched = True
    else:
        unbatched = False

    B, T, D = z.shape
    
    # When z=0, we want h[t] = h[t-1] + h_bar[t]
    # When z=1, we want h[t] = h_bar[t]
    # For values in between, we interpolate
    
    # First compute what the state would be if z=0 everywhere
    # This is just cumsum of h_bar plus h0
    h_cumsum = torch.cumsum(h_bar, dim=1) + h0.unsqueeze(1)
    
    # Now compute what the state would be if z=1 everywhere
    # This is just h_bar
    h_reset = h_bar
    
    # Interpolate between the two based on z
    h = (1 - z) * h_cumsum + z * h_reset
    
    if unbatched:
        h = h.squeeze(0)
    
    return h


class MinGRULayer(torch.nn.Module):
    def __init__(self, input_dim, state_dim, out_features, num_points):
        super(MinGRULayer, self).__init__()

        self.z_layer = AdaptivePiecewiseLinear(num_inputs=input_dim, num_outputs=state_dim, num_points=num_points)
        self.h_layer = AdaptivePiecewiseLinear(num_inputs=input_dim, num_outputs=state_dim,num_points=num_points)
        self.out_layer = AdaptivePiecewiseLinear(num_inputs=state_dim, num_outputs=out_features, num_points=num_points)
        self.hidden_size = state_dim
    
    def forward(self, x, h):
        """
        Forward pass using prefix sum.

        Args:
            x: Input tensor of shape (T, input_dim) or (B, T, input_dim)
            h: Initial hidden state of shape (state_dim,) or (B, state_dim)
               where B is batch size, T is sequence length

        Returns:
            Tuple[Tensor, Tensor]: (output tensor, hidden states)
            - If unbatched: shapes ((T, out_features), (T, state_dim))
            - If batched: shapes ((B, T, out_features), (B, T, state_dim))
        """
        
            
        # Handle batched case
        if x.dim() == 3:
            B, T, _ = x.shape
            # Reshape for linear layers
            x_reshaped = x.reshape(-1, x.size(-1))
            h_bar = self.h_layer(x_reshaped).reshape(B, T, -1)
            zt = torch.sigmoid(self.z_layer(x_reshaped)).reshape(B, T, -1)
        elif x.dim() == 2:
            # Add feature dimension if not present
            x = x.unsqueeze(-1)
            B, T = x.shape[:2]
            x_reshaped = x.reshape(-1, x.size(-1))
            h_bar = self.h_layer(x_reshaped).reshape(B, T, -1)
            zt = torch.sigmoid(self.z_layer(x_reshaped)).reshape(B, T, -1)
        else:
            h_bar = torch.relu(self.h_layer(x))
            zt = torch.sigmoid(self.z_layer(x))
        
        ht = prefix_sum_hidden_states(zt, h_bar, h)
        
        # Reshape ht for output layer
        if ht.dim() == 3:
            B, T, _ = ht.shape
            ht_reshaped = ht.reshape(-1, ht.size(-1))
            y = self.out_layer(ht_reshaped).reshape(B, T, -1)
        else:
            y = self.out_layer(ht)
            
        return y, ht


class MinGRUStack(torch.nn.Module):
    def __init__(self, input_dim, state_dim, out_features, layers, num_points):
        super(MinGRUStack, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(
            MinGRULayer(input_dim=input_dim, state_dim=state_dim, out_features=state_dim, num_points=num_points)
        )
        for _ in range(layers - 1):
            self.layers.append(
                MinGRULayer(input_dim=state_dim, state_dim=state_dim, out_features=state_dim, num_points=num_points)
            )
        self.output_layer = torch.nn.Linear(in_features=state_dim, out_features=out_features, bias=True)
        self.output_layer = AdaptivePiecewiseLinear(num_inputs=state_dim,num_outputs=out_features, num_points=num_points)
        self.state_dim = state_dim

    def forward(self, x, h=None):
        """
        Forward pass through the GRU stack.
        
        Args:
            x: Input tensor of shape (B, T, input_dim) or (T, input_dim)
            h: List of initial hidden states for each layer, or None
            
        Returns:
            output: Output tensor after final linear layer
            hidden_states: List of final hidden states from each GRU layer
        """
        if h is None:
            if x.dim() == 3:
                B = x.size(0)
                h = [torch.zeros(B, self.state_dim, device=x.device) for _ in range(len(self.layers))]
            else:
                h = [torch.zeros(self.state_dim, device=x.device) for _ in range(len(self.layers))]
        elif isinstance(h, list):
            # Already a list of hidden states, keep as is
            pass
        else:
            # Convert single tensor to list of hidden states
            h = [h_i.squeeze(1) if h_i.dim() == 3 else h_i for h_i in h]
        
        hidden_states = []
        current_x = x
        
        # Process through GRU layers
        for i, layer in enumerate(self.layers):
            current_x, new_h = layer(current_x, h[i])
            # Ensure new_h has correct shape [B, hidden] not [B, 1, hidden]
            if new_h.dim() == 3:
                new_h = new_h.squeeze(1)
            hidden_states.append(new_h)
        
        # Apply final output layer
        if current_x.dim() == 3:
            B, T, _ = current_x.shape
            current_x_reshaped = current_x.reshape(-1, current_x.size(-1))
            output = self.output_layer(current_x_reshaped).reshape(B, T, -1)
        else:
            output = self.output_layer(current_x)
        
        return output, hidden_states
