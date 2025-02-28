import torch
from non_uniform_piecewise_layers.adaptive_piecewise_linear import AdaptivePiecewiseLinear
from non_uniform_piecewise_layers.utils import norm_type, max_abs
from typing import Optional
from torch import Tensor

def max_abs_normalization_mingru(x: Tensor, eps: float = 1e-6):
    shape = x.shape
    xn = x.reshape(shape[0]*shape[1], -1)
    norm = xn / (max_abs(xn) + eps)
    return norm.reshape(shape)

def solve_recurrence(a, b, h0):
    """
    Computes h[t] = a[t] * h[t-1] + b[t] for t >= 0 in a vectorized manner.

    The closed-form solution is reformulated using logarithms to avoid division:

        h[t] = exp(log_A[t]) * (h0 + sum_{k=0}^{t} (b[k] * exp(log_S[k])))

    Args:
        a (torch.Tensor): Multiplicative coefficients of shape (T,)
        b (torch.Tensor): Additive coefficients of shape (T,)
        h0 (float or torch.Tensor): Initial condition h0

    Returns:
        torch.Tensor: Computed sequence h of shape (T,)
    """
    # Compute the logarithm of the cumulative product: log_A[t] = log(a[0]) + ... + log(a[t])
    log_a = torch.log(a)
    log_A = torch.cumsum(log_a, dim=1)
    
    # Compute the scaled b without division: log_S[k] = -log_A[k]
    log_S = -log_A

    # Compute the cumulative sum in the log domain: S[t] = sum_{k=0}^{t} (b[k] * exp(log_S[k]))
    exp_log_S = torch.exp(log_S)
    b_scaled = b * exp_log_S
    S = torch.cumsum(b_scaled, dim=1)
    
    # Final sequence: h[t] = exp(log_A[t]) * (h0 + S[t])
    exp_log_A = torch.exp(log_A)
    h = exp_log_A * (h0.unsqueeze(1) + S)

    return h



def solve_recurrence_unstable(a, b, h0):
    """
    Computes h[t] = a[t] * h[t-1] + b[t] for t >= 0 in a vectorized manner.
    
    The closed-form solution is:
    
        h[t] = (prod_{i=0}^{t} a[i]) * ( h0 + sum_{k=0}^{t} (b[k] / (prod_{i=0}^{k} a[i])) )
    
    Args:
        a (torch.Tensor): Multiplicative coefficients of shape (T,)
        b (torch.Tensor): Additive coefficients of shape (T,)
        h0 (float or torch.Tensor): Initial condition h0
        
    Returns:
        torch.Tensor: Computed sequence h of shape (T,)
    """
    # Forward cumulative product: A[t] = a[0]*...*a[t]
    A = torch.cumprod(a, dim=1)
    #print('A',A)
    # Scale b by dividing each b[k] by A[k]
    b_scaled = b / A
    #print('b_scaled', b_scaled)
    # Compute cumulative sum: S[t] = sum_{k=0}^{t} (b[k] / A[k])
    S = torch.cumsum(b_scaled, dim=1)
    
    # Final sequence: h[t] = A[t] * (h0 + S[t])
    h = A * (h0.unsqueeze(1) + S)
    
    return h


def prefix_sum_hidden_states(z, h_bar, h0):
    a = (1-z)
    b=z*h_bar
    #ans = solve_recurrence_unstable(a,b,h0)
    ans = solve_recurrence_unstable(a,b,h0)

    return ans


class MinGRULayer(torch.nn.Module):
    def __init__(self, input_dim, state_dim, out_features, num_points, position_init="uniform"):
        super(MinGRULayer, self).__init__()

        # Register layers as submodules
        self.add_module('z_layer', AdaptivePiecewiseLinear(num_inputs=input_dim, num_outputs=state_dim, num_points=num_points, position_init=position_init))
        self.add_module('h_layer', AdaptivePiecewiseLinear(num_inputs=input_dim, num_outputs=state_dim, num_points=num_points, position_init=position_init))
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

        ht = max_abs_normalization_mingru(ht)

        return ht

    def remove_add(self, x, h):
        """
        Remove the smoothest point and add a new point at the specified location
        for each layer in the MinGRU cell.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            h (torch.Tensor): Hidden state tensor of shape (batch_size, state_dim)
            
        Returns:
            bool: True if points were successfully removed and added in all layers,
                False otherwise.
        """
        with torch.no_grad():
            # Forward pass to get intermediate values
            x_reshaped = x.reshape(-1, x.size(-1))
            h_bar = self.h_layer(x_reshaped)
            zt = self.z_layer(x_reshaped)
            
            # Try removing and adding points in each layer
            success = True
            success &= self.z_layer.remove_add(x_reshaped)
            success &= self.h_layer.remove_add(x_reshaped)
            
            return success

    def insert_nearby_point(self, x, h):
        """
        Insert nearby points in each AdaptivePiecewiseLinear layer in the MinGRU cell.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            h (torch.Tensor): Hidden state tensor of shape (batch_size, state_dim)
            
        Returns:
            bool: True if points were successfully inserted in any layer,
                False otherwise.
        """
        with torch.no_grad():
            # Forward pass to get intermediate values
            x_reshaped = x.reshape(-1, x.size(-1))
            h_bar = self.h_layer(x_reshaped)
            zt = self.z_layer(x_reshaped)
            
            # Try inserting nearby points in each layer
            success = True
            success &= self.z_layer.insert_nearby_point(x_reshaped)
            success &= self.h_layer.insert_nearby_point(x_reshaped)
            
            return success

    def move_smoothest(self, weighted:bool=True):
        """
        Remove the point with the smallest removal error (smoothest point) and insert
        a new point randomly to the left or right of the point that would cause the
        largest error when removed for each AdaptivePiecewiseLinear layer in the MinGRU cell.
        
        Returns:
            bool: True if points were successfully moved in all layers, False otherwise.
        """
        with torch.no_grad():
            # Try moving the smoothest point in each layer
            success = True
            success &= self.z_layer.move_smoothest(weighted=weighted)
            success &= self.h_layer.move_smoothest(weighted=weighted)
            
            return success


class MinGRUStack(torch.nn.Module):
    def __init__(self, input_dim, state_dim, out_features, layers, num_points, position_init="uniform"):
        super(MinGRUStack, self).__init__()
        self.layers = torch.nn.ModuleList()

        self.layers.append(
            MinGRULayer(input_dim=input_dim, state_dim=state_dim, out_features=state_dim, num_points=num_points, position_init=position_init)
        )
        for _ in range(layers - 1):
            self.layers.append(
                MinGRULayer(input_dim=state_dim, state_dim=state_dim, out_features=state_dim, num_points=num_points, position_init=position_init)
            )

        # Register output layer as a named module
        self.add_module('output_layer', AdaptivePiecewiseLinear(num_inputs=state_dim, num_outputs=out_features, num_points=num_points, position_init=position_init))
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

        return output, tuple(hidden_states)

    def remove_add(self, x, h=None):
        """
        Remove the smoothest point and add a new point at the specified location
        for each layer in the MinGRU stack.

        Args:
            x (torch.Tensor): Reference point with shape (batch_size, T, input_dim)
            h (List[torch.Tensor], optional): List of initial hidden states for each layer

        Returns:
            bool: True if points were successfully removed and added in all layers,
                False otherwise.
        """
        with torch.no_grad():
            if h is None:
                B = x.size(0)
                h = [torch.zeros(B, self.state_dim, device=x.device) for _ in range(len(self.layers))]

            # Process through GRU layers
            success = True
            current_x = x
            for i, layer in enumerate(self.layers):
                new_h = layer(current_x, h[i])
                success &= layer.remove_add(current_x, h[i])
                current_x = new_h
            
            # Process output layer
            B, T, _ = current_x.shape
            current_x_reshaped = current_x.reshape(-1, current_x.size(-1))
            success &= self.output_layer.remove_add(current_x_reshaped)
        
        return success

    def insert_nearby_point(self, x, h=None):
        """
        Insert nearby points in each MinGRULayer and the output layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, T, input_dim)
            h (List[torch.Tensor], optional): List of initial hidden states for each layer
            
        Returns:
            bool: True if points were successfully inserted in any layer,
                False otherwise.
        """
        with torch.no_grad():
            if h is None:
                B = x.size(0)
                h = [torch.zeros(B, self.state_dim, device=x.device) for _ in range(len(self.layers))]
            
            # Process through GRU layers
            success = True
            current_x = x
            for i, layer in enumerate(self.layers):
                success &= layer.insert_nearby_point(current_x, h[i])
                new_h = layer(current_x, h[i])
                current_x = new_h
            
            # Process output layer
            B, T, _ = current_x.shape
            current_x_reshaped = current_x.reshape(-1, current_x.size(-1))
            success &= self.output_layer.insert_nearby_point(current_x_reshaped)
            
            return success

    def move_smoothest(self, weighted:bool=True):
        """
        Remove the point with the smallest removal error (smoothest point) and insert
        a new point randomly to the left or right of the point that would cause the
        largest error when removed for each layer in the MinGRU stack.
        
        Returns:
            bool: True if points were successfully moved in all layers, False otherwise.
        """
        with torch.no_grad():
            # Process through GRU layers
            success = True
            for layer in self.layers:
                success &= layer.move_smoothest(weighted=weighted)
            
            # Process output layer
            success &= self.output_layer.move_smoothest(weighted=weighted)
            
            return success

    def get_activation(self, name):
        def hook(model, input, output):
            model.activations = output.detach()
        return hook