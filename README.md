# Dynamic Non-Uniform Piecewise Linear Layers

## Example adding nodes

![Example of function approximation](readme-images/final_approximation.png)
![Training progress animation](readme-images/training_progress_loop.gif)

## Example moving nodes (square wave)

![Dynamic square wave](readme-images/dynamic_square_wave_loop.gif) 

## Example moving nodes (circle)

![Dynamic Circle](readme-images/progress_loop.gif)

### Layer weights for the circle

![Dynamic Circle Weights](readme-images/weights_loop.gif)

A PyTorch implementation of non-uniform piecewise linear layers. These layers can learn arbitrary continuous piecewise linear functions, where both the positions (x-coordinates) and values (y-coordinates) of the control points are learned parameters.

I typically run these in either dynamic mode, i.e, adding nodes or adaptively where the nodes are moved, but conserving total number of nodes.

this is a work in progress

### Function Approximation Example

See `examples/sine_fitting.py` for a complete example of approximating a complex function using the non-uniform piecewise linear layer.

## Square Wave
Non default example
```
python examples/dynamic_square_wave.py training.adapt=move model.num_points=20 training.refine_every_n_epochs=10 data.num_points=100
```

## MNIST
Running with and moving nodes with varrying number of points. You can run with larger learning_rate, to get faster results
```
python examples/mnist_classification.py -m model_type=adaptive epochs=100 move_nodes=True,False num_points=10 learning_rate=1e-4
```

## Shakespeare
Approaching good results with things like this
```
python examples/shakespeare_generation.py -m training.learning_rate=1e-3 training.num_epochs=20 training.move_every_n_batches=200 model.hidden_size=32,64 model.num_points=32 training.batch_size=128
```
small memory machine
```
python examples/shakespeare_generation.py -m training.learning_rate=1e-3 training.num_epochs=20 training.move_every_n_batches=50 model.hidden_size=16 model.num_points=32 training.batch_size=64 training.adapt=move
```

## 3D Implicit Representation
This one is pretty solid
```
python examples/implicit_3d.py mesh_resolution=100 learning_rate=1e-5 hidden_layers=[40, 40]
```
rendering after run
```
python examples/implicit_3d.py render_only=true model_path=/path/to/model.pt high_res_resolution=256
```
different output name
```
python examples/implicit_3d.py render_high_res=true render_output_file="my_render.png"
```

## Running visualization tests
use the -v to write data to file
```
pytest tests/test_visualization.py -v
```

## Interesting Papers

[Deep Networks Always Grok and Here's Why](https://arxiv.org/pdf/2402.15555)

Papers below are not currently used in this project, but would be interesting to investigate in the future
[Oja’s plasticity rule overcomes several challenges of training neural networks under biological constraints](https://arxiv.org/html/2408.08408)

[Continual Learning with Hebbian Plasticity in Sparse and Predictive Coding Networks: A Survey and Perspective](https://arxiv.org/abs/2407.17305)