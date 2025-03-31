import torch
import torch.nn as nn
import time
import sys
import os
import argparse

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from non_uniform_piecewise_layers.custom_positions_conv import CustomPositionsPiecewiseConv2d
from non_uniform_piecewise_layers.adaptive_piecewise_conv import AdaptivePiecewiseConv2d

def time_layer_forward(layer, x, device, num_warmup=10, num_runs=100):
    """Times the forward pass of a layer."""
    layer.to(device)
    x = x.to(device)
    
    # Warm-up runs
    for _ in range(num_warmup):
        _ = layer(x)
        
    # Synchronization for accurate timing (especially on GPU)
    if device == torch.device("cuda"):
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    
    # Timed runs
    for _ in range(num_runs):
        _ = layer(x)
        
    # Synchronization
    if device == torch.device("cuda"):
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time

def main(args):
    # --- Parameters ---
    batch_size = args.batch_size
    in_channels = args.in_channels
    out_channels = args.out_channels
    height = args.height
    width = args.width
    kernel_size = args.kernel_size
    num_points = args.num_points
    stride = args.stride
    padding = args.padding
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running performance comparison on device: {device}")
    print(f"Parameters:")
    print(f"  Batch Size: {batch_size}, In Channels: {in_channels}, Out Channels: {out_channels}")
    print(f"  Height: {height}, Width: {width}, Kernel Size: {kernel_size}")
    print(f"  Num Points: {num_points}, Stride: {stride}, Padding: {padding}")
    print("---")

    # --- Input Data ---
    x = torch.randn(batch_size, in_channels, height, width)

    # --- Layers ---
    print("Initializing layers...")
    try:
        custom_layer = CustomPositionsPiecewiseConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_points=num_points,
            stride=stride,
            padding=padding,
            position_init="uniform"
        )
        
        original_layer = AdaptivePiecewiseConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_points=num_points,
            stride=stride,
            padding=padding,
            position_init="uniform"
        )
    except Exception as e:
        print(f"Error initializing layers: {e}")
        return

    # --- Timing ---
    print("Timing CustomPositionsPiecewiseConv2d...")
    try:
        custom_time = time_layer_forward(custom_layer, x, device, num_warmup=args.warmup, num_runs=args.runs)
        print(f"  Average forward pass time: {custom_time * 1000:.4f} ms")
    except Exception as e:
        print(f"  Error during timing: {e}")
        custom_time = float('inf')

    print("Timing AdaptivePiecewiseConv2d...")
    try:
        original_time = time_layer_forward(original_layer, x, device, num_warmup=args.warmup, num_runs=args.runs)
        print(f"  Average forward pass time: {original_time * 1000:.4f} ms")
    except Exception as e:
        print(f"  Error during timing: {e}")
        original_time = float('inf')

    # --- Results ---
    print("---")
    if custom_time != float('inf') and original_time != float('inf') and original_time > 0:
        speedup = original_time / custom_time
        print(f"Custom positions layer is {speedup:.2f}x faster than the original layer.")
    elif custom_time == float('inf') or original_time == float('inf'):
        print("Could not complete comparison due to errors.")
    else:
         print("Comparison complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare performance of AdaptivePiecewiseConv2d implementations.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for input tensor")
    parser.add_argument("--in-channels", type=int, default=8, help="Number of input channels")
    parser.add_argument("--out-channels", type=int, default=16, help="Number of output channels")
    parser.add_argument("--height", type=int, default=16, help="Height of input tensor")
    parser.add_argument("--width", type=int, default=16, help="Width of input tensor")
    parser.add_argument("--kernel-size", type=int, default=3, help="Kernel size for convolution")
    parser.add_argument("--num-points", type=int, default=5, help="Number of points for piecewise functions")
    parser.add_argument("--stride", type=int, default=1, help="Stride for convolution")
    parser.add_argument("--padding", type=int, default=1, help="Padding for convolution")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=500, help="Number of timed runs")
    
    args = parser.parse_args()
    main(args)
