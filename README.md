# Event-Motion-Tracker

## Description:
A GPU-accelerated event-based vision pipeline in OpenCL inspired by Dynamic Vision Sensors. The pipeline converts standard camera frames into asynchronous temporal brightness change events, accumulates them into decaying event surfaces, and tracks relatively high-speed motion centroids/events.

## Programming Environment:
This repository should have everything needed to run the project, there is a change to the .devcontainer/gpu-adreno folder to include a Dockerfile and edits to the devcontainer.json file to allow OpenCV to work on the RB3. The output videos and images are stored on a separate output folder only on GitHub for organization purposes. When running the program it will be stored in the root of the directory.

## GitHub Structure

```
rb3-event-motion-tracker/
├── Makefile
├── README.md
├── input video/
│   ├── Video.mov
|   ├── Video.mp4
├── output/
│   ├── event_layer1.png
|   ├── event_layer2.png
|   ├── event_layer3.png
|   ├── event_out.mp4
|   ├── video_out.mp4
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
[Input Video]
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
 [Tracked Centroid on Object]
       │  (Red Circle on Object)
       ▼
[Event Video and Centroid Video]
```

