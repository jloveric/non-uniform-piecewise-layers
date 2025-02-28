# Non-Uniform Piecewise Linear Layers

![Example of function approximation](examples/final_approximation.png)

A PyTorch implementation of non-uniform piecewise linear layers. These layers can learn arbitrary continuous piecewise linear functions, where both the positions (x-coordinates) and values (y-coordinates) of the control points are learned parameters.

## Installation

```bash
pip install non-uniform-piecewise-layers
```

Or install from source:

```bash
git clone https://github.com/jloveric/non-uniform-piecewise-layers.git
cd non-uniform-piecewise-layers
pip install -e .
```

## Usage

### Basic Example

```python
import torch
from non_uniform_piecewise_layers import NonUniformPiecewiseLinear

# Create a layer with 2 inputs, 3 outputs, and 10 control points per function
layer = NonUniformPiecewiseLinear(
    num_inputs=2,
    num_outputs=3,
    num_points=10
)

# Forward pass
batch_size = 32
x = torch.randn(batch_size, 2)  # Input tensor
y = layer(x)  # Output shape: (batch_size, 3)

# Enforce monotonicity of control points (optional)
layer.enforce_monotonic()
```

### Function Approximation Example

See `examples/sine_fitting.py` for a complete example of approximating a complex function using the non-uniform piecewise linear layer. The example includes:

- Training setup with PyTorch
- Loss function and optimization
- Visualization of results
- Control point position monitoring

## Layer Architecture

The layer consists of the following learnable parameters:

- `positions`: Control point x-coordinates with shape (num_inputs, num_outputs, num_points)
- `values`: Control point y-coordinates with shape (num_inputs, num_outputs, num_points)

For each input-output pair, the layer learns a separate piecewise linear function defined by `num_points` control points. The forward pass performs efficient linear interpolation between these points.

## Shakespeare
Approaching good results with things like this
```
python examples/shakespeare_generation.py -m training.learning_rate=1e-3 training.num_epochs=20 training.remove_add_every_n_batches=200 model.hidden_size=32,64 model.num_points=32 training.batch_size=128
```
small memory machine
```
python examples/shakespeare_generation.py -m training.learning_rate=1e-3 training.num_epochs=20 training.remove_add_every_n_batches=50 model.hidden_size=16 model.num_points=32 training.batch_size=64 training.adapt=move
```

## Running visualization tests
use the -v to write data to file
```
pytest tests/test_visualization.py -v
```