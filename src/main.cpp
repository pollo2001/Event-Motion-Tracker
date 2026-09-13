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


#include <stdio.h>
#include <stdlib.h>

#include "device.h"
#include "kernel.h"
#include "matrix.h"
#include <opencv2/opencv.hpp>
#include <iostream>

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

#define KERNEL_PATH "kernel.cl"


int main() {

    cv::VideoCapture cap("video/Video.mov"); // find video

    if (!cap.isOpened()) {
        std::cerr << "Error: could not open video\n";
        return -1;
    }

    cv::Mat frame;

    while (cap.read(frame)) {
        std::cout << "Read frame successfully\n";

    }

    std::cout << "FPS: "
          << cap.get(cv::CAP_PROP_FPS)
          << "\n";

    std::cout << "Frame count: "
            << cap.get(cv::CAP_PROP_FRAME_COUNT)
            << "\n";

    std::cout << "Duration: "
            << cap.get(cv::CAP_PROP_FRAME_COUNT) /
                cap.get(cv::CAP_PROP_FPS)
            << " seconds\n";

    cap.release();
    return 0;
}
