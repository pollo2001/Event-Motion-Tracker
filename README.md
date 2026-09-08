# Event-Motion-Tracker

## Description:
Event-based motion tracking in OpenCL using the RB3.

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

##Layer Structure
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

