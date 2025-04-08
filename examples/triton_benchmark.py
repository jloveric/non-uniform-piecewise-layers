import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path to import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from non_uniform_piecewise_layers.adaptive_piecewise_conv import AdaptivePiecewiseConv2d
from non_uniform_piecewise_layers.adaptive_piecewise_mlp import AdaptivePiecewiseMLP
from non_uniform_piecewise_layers.triton_adaptive_piecewise_conv import (
    TritonAdaptivePiecewiseConv2d,
    TritonAdaptivePiecewiseMLP,
    TritonAdaptivePiecewiseLinear
)
from non_uniform_piecewise_layers.optimized_triton_adaptive_piecewise_conv import (
    OptimizedTritonAdaptivePiecewiseConv2d
)
from non_uniform_piecewise_layers.hybrid_triton_adaptive_piecewise_conv import (
    HybridTritonAdaptivePiecewiseConv2d
)
from non_uniform_piecewise_layers.fast_triton_adaptive_piecewise_conv import (
    FastTritonAdaptivePiecewiseConv2d
)


def benchmark_conv2d(batch_sizes, device="cuda"):
    """
    Benchmark the performance of AdaptivePiecewiseConv2d vs TritonAdaptivePiecewiseConv2d
    and OptimizedTritonAdaptivePiecewiseConv2d
    """
    # Parameters
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    num_points = 5
    input_size = 32
    
    # Create models
    pytorch_model = AdaptivePiecewiseConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        num_points=num_points
    ).to(device)
    
    triton_model = TritonAdaptivePiecewiseConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        num_points=num_points
    ).to(device)
    
    optimized_triton_model = OptimizedTritonAdaptivePiecewiseConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        num_points=num_points
    ).to(device)
    
    hybrid_triton_model = HybridTritonAdaptivePiecewiseConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        num_points=num_points
    ).to(device)
    
    fast_triton_model = FastTritonAdaptivePiecewiseConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        num_points=num_points
    ).to(device)
    
    # Copy weights from PyTorch model to Triton models for fair comparison
    triton_model.values.data = pytorch_model.piecewise.values.data.permute(1, 0, 2).reshape(
        out_channels, in_channels, kernel_size, kernel_size, num_points
    )
    
    optimized_triton_model.values.data = pytorch_model.piecewise.values.data.permute(1, 0, 2).reshape(
        out_channels, in_channels, kernel_size, kernel_size, num_points
    )
    
    hybrid_triton_model.values.data = pytorch_model.piecewise.values.data.permute(1, 0, 2).reshape(
        out_channels, in_channels, kernel_size, kernel_size, num_points
    )
    
    fast_triton_model.values.data = pytorch_model.piecewise.values.data.permute(1, 0, 2).reshape(
        out_channels, in_channels, kernel_size, kernel_size, num_points
    )
    
    # Benchmark results
    pytorch_times = []
    triton_times = []
    optimized_triton_times = []
    hybrid_triton_times = []
    fast_triton_times = []
    triton_speedups = []
    optimized_speedups = []
    hybrid_speedups = []
    fast_speedups = []
    
    # Warmup
    for _ in range(10):
        x = torch.randn(4, in_channels, input_size, input_size, device=device)
        pytorch_model(x)
        triton_model(x)
        optimized_triton_model(x)
        hybrid_triton_model(x)
        fast_triton_model(x)
    
    # Benchmark
    for batch_size in batch_sizes:
        # Create input
        x = torch.randn(batch_size, in_channels, input_size, input_size, device=device)
        
        # PyTorch model
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            pytorch_model(x)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start) / 100
        pytorch_times.append(pytorch_time * 1000)  # Convert to ms
        
        # Triton model
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            triton_model(x)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / 100
        triton_times.append(triton_time * 1000)  # Convert to ms
        
        # Optimized Triton model
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            optimized_triton_model(x)
        torch.cuda.synchronize()
        optimized_triton_time = (time.time() - start) / 100
        optimized_triton_times.append(optimized_triton_time * 1000)  # Convert to ms
        
        # Hybrid Triton model
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            hybrid_triton_model(x)
        torch.cuda.synchronize()
        hybrid_triton_time = (time.time() - start) / 100
        hybrid_triton_times.append(hybrid_triton_time * 1000)  # Convert to ms
        
        # Fast Triton model
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            fast_triton_model(x)
        torch.cuda.synchronize()
        fast_triton_time = (time.time() - start) / 100
        fast_triton_times.append(fast_triton_time * 1000)  # Convert to ms
        
        # Calculate speedups
        triton_speedup = pytorch_time / triton_time
        optimized_speedup = pytorch_time / optimized_triton_time
        hybrid_speedup = pytorch_time / hybrid_triton_time
        fast_speedup = pytorch_time / fast_triton_time
        triton_speedups.append(triton_speedup)
        optimized_speedups.append(optimized_speedup)
        hybrid_speedups.append(hybrid_speedup)
        fast_speedups.append(fast_speedup)
        
        print(f"Batch size: {batch_size}")
        print(f"  PyTorch time: {pytorch_time * 1000:.2f} ms")
        print(f"  Triton time: {triton_time * 1000:.2f} ms")
        print(f"  Optimized Triton time: {optimized_triton_time * 1000:.2f} ms")
        print(f"  Hybrid Triton time: {hybrid_triton_time * 1000:.2f} ms")
        print(f"  Fast Triton time: {fast_triton_time * 1000:.2f} ms")
        print(f"  Triton Speedup: {triton_speedup:.2f}x")
        print(f"  Optimized Speedup: {optimized_speedup:.2f}x")
        print(f"  Hybrid Speedup: {hybrid_speedup:.2f}x")
        print(f"  Fast Speedup: {fast_speedup:.2f}x")
        
        # Verify outputs are similar
        with torch.no_grad():
            pytorch_output = pytorch_model(x)
            triton_output = triton_model(x)
            max_diff = torch.max(torch.abs(pytorch_output - triton_output)).item()
            print(f"  Max difference: {max_diff:.6f}")
    
    return pytorch_times, triton_times, speedups


def benchmark_mlp(input_sizes, device="cuda"):
    """
    Benchmark the performance of AdaptivePiecewiseMLP vs TritonAdaptivePiecewiseMLP
    """
    # Parameters
    batch_size = 128
    width = [64, 128, 64, 32]
    num_points = 5
    
    # Create models
    pytorch_model = AdaptivePiecewiseMLP(
        width=width,
        num_points=num_points
    ).to(device)
    
    triton_model = TritonAdaptivePiecewiseMLP(
        width=width,
        num_points=num_points
    ).to(device)
    
    # Copy weights for fair comparison
    for i, (pytorch_layer, triton_layer) in enumerate(zip(pytorch_model.layers, triton_model.layers)):
        triton_layer.values.data = pytorch_layer.values.data
    
    # Benchmark results
    pytorch_times = []
    triton_times = []
    speedups = []
    
    # Warmup
    for _ in range(10):
        x = torch.randn(batch_size, width[0], device=device)
        pytorch_model(x)
        triton_model(x)
    
    # Benchmark
    for input_size in input_sizes:
        # Adjust width for this test
        adjusted_width = [input_size] + width[1:]
        
        # Create new models with adjusted width
        pytorch_model = AdaptivePiecewiseMLP(
            width=adjusted_width,
            num_points=num_points
        ).to(device)
        
        triton_model = TritonAdaptivePiecewiseMLP(
            width=adjusted_width,
            num_points=num_points
        ).to(device)
        
        # Copy weights for fair comparison
        for i, (pytorch_layer, triton_layer) in enumerate(zip(pytorch_model.layers, triton_model.layers)):
            triton_layer.values.data = pytorch_layer.values.data
        
        # Create input
        x = torch.randn(batch_size, input_size, device=device)
        
        # PyTorch model
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            pytorch_model(x)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start) / 100
        pytorch_times.append(pytorch_time * 1000)  # Convert to ms
        
        # Triton model
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            triton_model(x)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / 100
        triton_times.append(triton_time * 1000)  # Convert to ms
        
        # Calculate speedup
        speedup = pytorch_time / triton_time
        speedups.append(speedup)
        
        print(f"Input size: {input_size}")
        print(f"  PyTorch time: {pytorch_time * 1000:.2f} ms")
        print(f"  Triton time: {triton_time * 1000:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Verify outputs are similar
        with torch.no_grad():
            pytorch_output = pytorch_model(x)
            triton_output = triton_model(x)
            max_diff = torch.max(torch.abs(pytorch_output - triton_output)).item()
            print(f"  Max difference: {max_diff:.6f}")
    
    return pytorch_times, triton_times, speedups


def benchmark_memory_usage(batch_sizes, device="cuda"):
    """
    Benchmark the memory usage of AdaptivePiecewiseConv2d vs TritonAdaptivePiecewiseConv2d
    """
    # Parameters
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    num_points = 5
    input_size = 32
    
    # Create models
    pytorch_model = AdaptivePiecewiseConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        num_points=num_points
    ).to(device)
    
    triton_model = TritonAdaptivePiecewiseConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        num_points=num_points
    ).to(device)
    
    # Memory usage results
    pytorch_memory = []
    triton_memory = []
    memory_savings = []
    
    for batch_size in batch_sizes:
        # Create input
        x = torch.randn(batch_size, in_channels, input_size, input_size, device=device)
        
        # PyTorch model - measure memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        pytorch_model(x)
        pytorch_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        pytorch_memory.append(pytorch_mem)
        
        # Triton model - measure memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        triton_model(x)
        triton_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        triton_memory.append(triton_mem)
        
        # Calculate memory savings
        memory_saving = (pytorch_mem - triton_mem) / pytorch_mem * 100  # percentage
        memory_savings.append(memory_saving)
        
        print(f"Batch size: {batch_size}")
        print(f"  PyTorch memory: {pytorch_mem:.2f} MB")
        print(f"  Triton memory: {triton_mem:.2f} MB")
        print(f"  Memory saving: {memory_saving:.2f}%")
    
    return pytorch_memory, triton_memory, memory_savings


def plot_results(batch_sizes, pytorch_times, triton_times, speedups, title):
    """
    Plot the benchmark results
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot times
    ax1.plot(batch_sizes, pytorch_times, 'o-', label='PyTorch')
    ax1.plot(batch_sizes, triton_times, 'o-', label='Triton')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Execution Time')
    ax1.legend()
    ax1.grid(True)
    
    # Plot speedups
    ax2.plot(batch_sizes, speedups, 'o-', color='green')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('Triton Speedup')
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.show()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device != "cuda":
        print("CUDA not available. Triton requires CUDA.")
        return
    
    print("Benchmarking Conv2d...")
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    pytorch_times, triton_times, speedups = benchmark_conv2d(batch_sizes, device)
    plot_results(batch_sizes, pytorch_times, triton_times, speedups, "AdaptivePiecewiseConv2d Benchmark")
    
    print("\nBenchmarking MLP...")
    input_sizes = [32, 64, 128, 256, 512, 1024]
    pytorch_times, triton_times, speedups = benchmark_mlp(input_sizes, device)
    plot_results(input_sizes, pytorch_times, triton_times, speedups, "AdaptivePiecewiseMLP Benchmark")
    
    print("\nBenchmarking Memory Usage...")
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    pytorch_memory, triton_memory, memory_savings = benchmark_memory_usage(batch_sizes, device)
    
    # Plot memory usage
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(batch_sizes, pytorch_memory, 'o-', label='PyTorch')
    ax1.plot(batch_sizes, triton_memory, 'o-', label='Triton')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Memory (MB)')
    ax1.set_title('Memory Usage')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(batch_sizes, memory_savings, 'o-', color='green')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Memory Saving (%)')
    ax2.set_title('Memory Savings')
    ax2.grid(True)
    
    plt.suptitle("Memory Usage Benchmark")
    plt.tight_layout()
    plt.savefig("memory_usage_benchmark.png")
    plt.show()


if __name__ == "__main__":
    main()
