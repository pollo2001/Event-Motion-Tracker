# Event-Motion-Tracker

## Description:
GPU-accelerated event-based vision pipeline in OpenCL inspired by Dynamic Vision Sensors. The pipeline converts standard camera frames into asynchronous temporal brightness change events, accumulates them into decaying event surfaces, and tracks relatively high-speed motion centroids/events.

## Project Structure

```
rb3-event-motion-tracker/
├── Makefile
├── README.md
├── src/
│   ├── main.cpp         # Captures frames, manages OpenCL queue, draws output
│   └── kernels.cl       # The 3 vision kernels (diff, decay, filter)
└── benchmarks/          # Optional: proof of baseline from Assignment 3
    └── block_mm_neon/
        ├── Makefile
        └── main.c
```

## Layer Structure
```
[Raw Camera Frame]
       │
       ▼
 ┌───────────┐
 │  Layer 1  │  Temporal Difference & Event Generation
 └─────┬─────┘
       │  (Binary / Polarity Event Stream)
       ▼
 ┌───────────┐
 │  Layer 2  │  Surface of Active Events (SAE) / Exponential Decay
 └─────┬─────┘
       │  (Fading 2D Motion Trail)
       ▼
 ┌───────────┐
 │  Layer 3  │  Spatial Filtering / Noise Rejection (or Centroid Reduction)
 └─────┬─────┘
       │  (Clean Target Coordinates)
       ▼
 [Tracked Centroid (X, Y)]
```

