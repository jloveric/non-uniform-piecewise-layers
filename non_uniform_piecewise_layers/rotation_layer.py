import torch

def fixed_rotation_layer(n: int, rotations: int = 2, normalize: bool = True):
    """
    Take n inputs and compute all the variations, n_i+n_j, n_i-n_j
    and create a layer that computes these with fixed weights. For
    n=2, and rotations=2 outputs [x, t, a(x+t), a(x-t)].  Returns a fixed
    linear rotation layer (one that is not updated by gradients)
    Args :
        - n: The number of inputs, would be 2 for (x, t)
        - rotations: Number of rotations to apply pair by based on the inputs. So
        for input [x, y] and rotations=3, rotations are [x, y,a*(x+t), a*(x-t) ]
        - normalize: If true, normalizes values to be between -1 and 1
    Returns :
        A tuple containing the rotation layer and the output width of the layer
    """

    if rotations < 1:
        raise ValueError(
            f"Rotations must be 1 or greater. 1 represents no additional rotations. Got rotations={rotations}"
        )

    combos = []
    for i in range(n):
        reg = [0] * n
        reg[i] = 1.0
        combos.append(reg)

        for j in range(i + 1, n):
            for r in range(1, rotations):
                # We need to add rotations from each of 2 quadrants
                temp = [0] * n

                theta = (math.pi / 2) * (r / rotations)
                rot_x = math.cos(theta)
                rot_y = math.sin(theta)
                norm_val = 1 if normalize is False else abs(rot_x) + abs(rot_y)

                # Add the line and the line orthogonal
                temp[i] += rot_x / norm_val
                temp[j] += rot_y / norm_val

                combos.append(temp)

                other = [0] * n
                other[i] += rot_y / norm_val
                other[j] += -rot_x / norm_val

                combos.append(other)

    # 2 inputs, 1 rotation -> 2 combos
    # 2 inputs, 2 rotations -> 4 combos
    # 2 inputs, 3 rotations -> 6 combos
    # 2 inputs, 4 rotations -> 8 combos
    output_width = n + n * (n - 1) * (rotations - 1)
    layer = torch.nn.Linear(n, output_width, bias=False)
    weights = torch.tensor(combos)
    layer.weight = torch.nn.Parameter(weights, requires_grad=False)
    return layer, output_width