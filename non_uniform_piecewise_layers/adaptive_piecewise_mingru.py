import torch
from non_uniform_piecewise_layers.adaptive_piecewise_linear import AdaptivePiecewiseLinear
from non_uniform_piecewise_layers.utils import norm_type
from typing import Optional


import torch

import torch

import torch

def solve_recurrence(a, b, h0):
    """
    TODO: Verify this with small vectors
    Computes h[t] = a[t] * h[t-1] + b[t] in a vectorized manner using torch.cumprod and torch.cumsum.
    
    Args:
        a (torch.Tensor): Multiplicative coefficients of shape (T,)
        b (torch.Tensor): Additive coefficients of shape (T,)
        h0 (float or torch.Tensor): Initial condition h[0]

    Returns:
        torch.Tensor: Computed sequence h of shape (T,)
    """
    # Compute cumulative product of a (shifted by one, with 1 prepended)
    B, T, V = a.shape
    A = torch.cumprod(a, dim=1)  # A[t] = prod(a[:t+1])

    ones_tensor = torch.ones((B, 1, V), device=A.device)

    # Reverse cumulative product of a (with 1 appended at the end)
    A_rev = torch.cat([ones_tensor, A[:,:-1,:]],dim=1)  # A_rev[t] = prod(a[t+1:T])

    # Compute the cumulative sum of weighted b
    B = torch.cumsum(A_rev * b, dim=1)  # Accumulate weighted contributions of b

    # Compute final h[t]
    h = A * h0.unsqueeze(1) + B

    return h


def prefix_sum_hidden_states(z, h_bar, h0):
    a = (1-z)
    b=z*h_bar
    ans = solve_recurrence(a,b,h0)
    return ans



# def prefix_sum_hidden_states(z, h_bar, h0):
#     """
#     Vectorized computation of hidden states using parallel scan.
#     Implements the recurrence: h[t+1] = (1-z[t])h[t] + z[t]*hbar[t]

#     Args:
#         z (Tensor): Update gate tensor of shape (B, T, D) or (T, D)
#         h_bar (Tensor): Candidate hidden states of shape (B, T, D) or (T, D)
#         h0 (Tensor): Initial hidden state of shape (B, D) or (D,)

#     Returns:
#         Tensor: Computed hidden states of shape (B, T, D) or (T, D)
#     """

#     B, T, D = z.shape
    
#     # When z=0, we want h[t] = h[t-1] + h_bar[t]
#     # When z=1, we want h[t] = h_bar[t]
#     # For values in between, we interpolate
    
#     # First compute what the state would be if z=0 everywhere
#     # This is just cumsum of h_bar plus h0
#     h_cumsum = torch.cumsum(h_bar, dim=1) + h0.unsqueeze(1)
#     # Now compute what the state would be if z=1 everywhere
#     # This is just h_bar
#     h_reset = h_bar
    
#     # Interpolate between the two based on z
#     h = (1 - z) * h_cumsum + z * h_reset
    
#     return h


class MinGRULayer(torch.nn.Module):
    def __init__(self, input_dim, state_dim, out_features, num_points):
        super(MinGRULayer, self).__init__()

        self.z_layer = AdaptivePiecewiseLinear(num_inputs=input_dim, num_outputs=state_dim, num_points=num_points)
        self.h_layer = AdaptivePiecewiseLinear(num_inputs=input_dim, num_outputs=state_dim,num_points=num_points)
        #self.out_layer = AdaptivePiecewiseLinear(num_inputs=state_dim, num_outputs=out_features, num_points=num_points)
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
        B, T, _ = x.shape
        # Reshape for linear layers
        x_reshaped = x.reshape(-1, x.size(-1))
        h_bar = self.h_layer(x_reshaped).reshape(B, T, -1)
        zt = torch.sigmoid(self.z_layer(x_reshaped)).reshape(B, T, -1)
        ht = prefix_sum_hidden_states(zt, h_bar, h)

        B, T, _ = ht.shape
        #ht_reshaped = ht.reshape(-1, ht.size(-1))
        #y = self.out_layer(ht_reshaped).reshape(B, T, -1)

        return ht

    def remove_add(self, x):
        """
        TODO: claude's implementation is obviously wrong fix!
        Remove the smoothest point and add a new point at the specified location
        for each adaptive piecewise linear layer.

        Args:
            x (torch.Tensor): Reference point with shape (batch_size, input_width)
                specifying where to add the new point.

        Returns:
            bool: True if points were successfully removed and added in all layers,
                False otherwise.
        """
        with torch.no_grad():
            # Try removing and adding points in each layer
            success = True
            success &= self.z_layer.remove_add(x)
            success &= self.h_layer.remove_add(x)
            success &= self.out_layer.remove_add(x)
        
        return success


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
            B = x.size(0)
            h = [torch.zeros(B, self.state_dim, device=x.device) for _ in range(len(self.layers))]
            
            """
            if x.dim() == 3:
                B = x.size(0)
                h = [torch.zeros(B, self.state_dim, device=x.device) for _ in range(len(self.layers))]
            else:
                h = [torch.zeros(self.state_dim, device=x.device) for _ in range(len(self.layers))]
            """
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
            new_h = layer(current_x, h[i])

            # Store only the last element
            hidden_states.append(new_h.squeeze(1))
            current_x = new_h

        # Apply final output layer
        """
        if current_x.dim() == 3:
            B, T, _ = current_x.shape
            current_x_reshaped = current_x.reshape(-1, current_x.size(-1))
            output = self.output_layer(current_x_reshaped).reshape(B, T, -1)
        else:
            output = self.output_layer(current_x)
        """
        B, T, _ = current_x.shape
        current_x_reshaped = current_x.reshape(-1, current_x.size(-1))
        output = self.output_layer(current_x_reshaped).reshape(B, T, -1)

        return output, hidden_states

    def remove_add(self, x):
        """
        TODO: Don't think claude's implementation is quite right, fix!
        Remove the smoothest point and add a new point at the specified location
        for each layer in the MinGRU stack.

        Args:
            x (torch.Tensor): Reference point with shape (batch_size, input_width)
                specifying where to add the new point.

        Returns:
            bool: True if points were successfully removed and added in all layers,
                False otherwise.
        """
        with torch.no_grad():
            # Forward pass to get intermediate values
            intermediate_x = [x]
            current_x = x
            h = None  # Start with zero hidden states
            for layer in self.layers:
                current_x, _ = layer(current_x, h)
                intermediate_x.append(current_x)
            
            # Try removing and adding points in each layer
            success = True
            for i, layer in enumerate(self.layers):
                success_ = layer.remove_add(intermediate_x[i])
                if not success_:
                    success = False
        
        return success
