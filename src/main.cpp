//main.cpp

/**
main.cpp
L!: Capture Frames
L2: Manage OpenCL queue
L3: Draw output
**/


//TO--DO
// Host CPU Orchestration:
// 1. Enqueue Layer 1 (current_frame, previous_frame -> event_map).
// 2. Enqueue Layer 2 (event_map, surface_history -> surface_history).
// 3. Enqueue Layer 3 (surface_history -> filtered_surface).
// 4. Read back filtered_surface to CPU host memory.
// 5. Compute Centroid:
//      cx = sum(x * intensity) / sum(intensity)
//      cy = sum(y * intensity) / sum(intensity)
// 6. Draw tracking circle at (cx, cy) on display frame.
// 7. Swap frame pointers: previous_frame = current_frame.

