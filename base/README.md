## Baseline Benchmarks & Relevance

To evaluate compute capabilities, we used the results of the matrix multiplication baselines from Assignment 3.

| Implementation | Total Execution Time (10 runs) | Isolated Compute Math |
| :--- | :--- | :--- |
| **ARM Neon Block (CPU)** | **1.645 ms** | 1.645 ms |
| **OpenCL Naive MM (GPU)** | 748.150 ms | 2.113 ms |
| **OpenCL Tiled MM (GPU)** | 777.642 ms | **1.425 ms** |

### Why This Drove Our Pipeline Design

* **The Problem:** Dense matrix multiplication ($C = A \cdot B$) is irrelevant to real-time event vision, which relies on element-wise frame differencing, temporal decay, and spatial reduction. However, testing it revealed a major hardware constraint: OpenCL introduces a flat **70–80 ms driver and dispatch tax** when allocating buffers and moving data across the bus. This is a constant overhead tax.
  
* **Why Pure Neon Isn't Enough:** While Neon SIMD runs entirely in CPU cache with zero dispatch latency, processing full-resolution frame-by-frame event surfaces entirely on the CPU steals cycles needed for camera capture, application logic, and streaming.
  
* **The Resulting Architecture:** Pure GPU compute is faster (1.425 ms vs. 1.645 ms). Therefore, our vision tracker chains all 3 layers (Differencing $\rightarrow$ Decay Surface $\rightarrow$ Filter) sequentially inside GPU memory (`cl_mem`). Buffers never bounce back to the host CPU between intermediate stages, effectively eliminating the bus-transfer penalty while freeing up the ARM Cortex cores.
