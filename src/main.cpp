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
#include <vector>

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

#define KERNEL_PATH "src/kernels.cl"

using namespace cv;
using namespace std;

Mat frame;
Mat frame_gray;
Mat previous_gray;

void OpenCLEventTracker(int frame_counter)
{
    // Load external OpenCL kernel code
    char *kernel_source = OclLoadKernel(KERNEL_PATH); // Load kernel source

    // Device input and output buffers
    cl_mem device_current_f, device_previous_f, device_event_map;
    cl_mem device_surface, device_output_surface;

    size_t global_item_size[2];
    size_t local_item_size;
    cl_int err;

    cl_device_id device_id;    // device ID
    cl_context context;        // context
    cl_command_queue queue;    // command queue
    cl_program program;        // program
    cl_kernel kernel1;          // kernel1
    cl_kernel kernel2;          // kernel2
    cl_kernel kernel3;          // kernel3

    const int threshold = 35;
    const int height = frame_gray.rows;
    const int width = frame_gray.cols;
    int event_map_entries = height * width;
    vector<int8_t> event_map_idx(event_map_entries);

    // Find platforms and devices
    OclPlatformProp *platforms = NULL;
    cl_uint num_platforms;

    err = OclFindPlatforms((const OclPlatformProp **)&platforms, &num_platforms);
    CHECK_ERR(err, "OclFindPlatforms");

    // Get the ID for the specified kind of device type.
    err = OclGetDeviceWithFallback(&device_id, OCL_DEVICE_TYPE);
    CHECK_ERR(err, "OclGetDeviceWithFallback");

    // Create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    CHECK_ERR(err, "clCreateContext");

    // Create a command queue
# if __APPLE__
    queue = clCreateCommandQueue(context, device_id, 0, &err);
#else
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
#endif
    CHECK_ERR(err, "clCreateCommandQueueWithProperties");

    // Create the program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, NULL, &err);
    CHECK_ERR(err, "clCreateProgramWithSource");

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    CHECK_ERR(err, "clBuildProgram");

    // Create the compute kernel in the program we wish to run
    kernel1 = clCreateKernel(program, "generate_events", &err);
    CHECK_ERR(err, "clCreateKernel");
    kernel2 = clCreateKernel(program, "generate_sae", &err);
    CHECK_ERR(err, "clCreateKernel");
    kernel3 = clCreateKernel(program, "generate_boxFilter", &err);
    CHECK_ERR(err, "clCreateKernel");

    //@@ Allocate GPU memory here
    size_t size_a = height * width;
    size_t size_b = height * width;
    size_t size_c = height * width;
    size_t size_d = height * width;
    size_t size_e = height * width;
    device_current_f = clCreateBuffer(context, CL_MEM_READ_ONLY, size_a * sizeof(uchar), NULL, &err );
    CHECK_ERR(err, "clCreateBuffer device_current_f");
    device_previous_f = clCreateBuffer(context, CL_MEM_READ_ONLY, size_b * sizeof(uchar), NULL, &err );
    CHECK_ERR(err, "clCreateBuffer device_previous_f");
    device_event_map = clCreateBuffer(context, CL_MEM_READ_WRITE, size_c * sizeof(char), NULL, &err );
    CHECK_ERR(err, "clCreateBuffer device_event_map");
    device_surface = clCreateBuffer(context, CL_MEM_READ_WRITE, size_d * sizeof(float), NULL, &err );
    CHECK_ERR(err, "clCreateBuffer device_surface");
    device_output_surface = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_e * sizeof(float), NULL, &err );
    CHECK_ERR(err, "clCreateBuffer device_output_surface");
    //@@ Copy memory to the GPU here
    err = clEnqueueWriteBuffer(queue, device_current_f, CL_TRUE, 0, size_a* sizeof(uchar), frame_gray.data, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer device_current_f");
    err = clEnqueueWriteBuffer(queue, device_previous_f, CL_TRUE, 0, size_b* sizeof(uchar), previous_gray.data, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer device_previous_f");
    //@@ define local and global work sizes
    global_item_size[0] = height;
    global_item_size[1] = width;
    local_item_size = 1;
    // Set the arguments to our compute kernel
    err = clSetKernelArg(kernel1, 0, sizeof(cl_mem), &device_current_f);
    CHECK_ERR(err, "clSetKernelArg 0");
    err |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &device_previous_f);
    CHECK_ERR(err, "clSetKernelArg 1");
    err |= clSetKernelArg(kernel1, 2, sizeof(cl_mem), &device_event_map);
    CHECK_ERR(err, "clSetKernelArg 2");
    err |= clSetKernelArg(kernel1, 3, sizeof(const int), &threshold);
    CHECK_ERR(err, "clSetKernelArg 3");
    err |= clSetKernelArg(kernel1, 4, sizeof(const int), &width);
    CHECK_ERR(err, "clSetKernelArg 4");
     err |= clSetKernelArg(kernel1, 5, sizeof(const int), &height);
    CHECK_ERR(err, "clSetKernelArg 5");

    //@@ Launch the GPU Kernel here
    err = clEnqueueNDRangeKernel(queue, kernel1, 2, NULL, global_item_size, NULL, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueNDRangeKernel");
    //@@ Copy the GPU memory back to the CPU here
    err = clEnqueueReadBuffer(queue, device_event_map, CL_TRUE, 0, size_c* sizeof(char), event_map_idx.data(), 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueReadBuffer");

int on_count = 0;
int off_count = 0;
int zero_count = 0;

for (int8_t e : event_map_idx)
{
    if (e == 1)
        on_count++;
    else if (e == -1)
        off_count++;
    else
        zero_count++;
}

cout << "ON: " << on_count
     << " OFF: " << off_count
     << " ZERO: " << zero_count
     << endl;

Mat event_image(height, width, CV_8UC1);

for (int i = 0; i < event_map_entries; i++)
{
    if (event_map_idx[i] == 1)
        event_image.data[i] = 255;   // ON = white

    else if (event_map_idx[i] == -1)
        event_image.data[i] = 128;   // OFF = gray

    else
        event_image.data[i] = 0;     // no event = black
}

if (frame_counter == 20)
{
    imwrite("event_debug_middle.png", event_image);
    cout << "Saved middle frame debug image" << endl;
}

    //@@ Free the GPU memory here
    clReleaseMemObject(device_current_f);
    clReleaseMemObject(device_previous_f);
    clReleaseMemObject(device_event_map);
    clReleaseMemObject(device_surface);
    clReleaseMemObject(device_output_surface);
    clReleaseProgram(program);
    clReleaseKernel(kernel1);
    clReleaseKernel(kernel2);
    clReleaseKernel(kernel3);


}


int main() {

    VideoCapture cap("video/Video.mov"); // find video

    int frame_counter = 0;

    if (!cap.isOpened()) {
        std::cerr << "Error: could not open video\n";
        return -1;
    }


    bool first_frame = true;

    while (cap.read(frame)) {
        std::cout << "Read frame successfully\n";

    cvtColor(frame,frame_gray, COLOR_BGR2GRAY);

    if(first_frame) //  only once used since first frame doesnt have a previous frame
    {
        previous_gray = frame_gray.clone(); // set a previous frame to current frame
        first_frame = false;
        continue;
    }
    frame_counter++;
    
    /*imshow("Grayscale Video Feed",frame_gray);

    char key = (char)waitKey(30);
    if(key == 27 || key == 'q')
    {
        break;
    }
    cout << "Gray frame: "
         << frame_gray.cols << "x"
         << frame_gray.rows
         << " channels="
         << frame_gray.channels()
         << endl;*/

    // Call your function.
    OpenCLEventTracker(frame_counter);

    previous_gray = frame_gray.clone(); // make sure previous frame is set after processing
}

    cout << "FPS: "
          << cap.get(cv::CAP_PROP_FPS)
          << "\n";

    cout << "Frame count: "
            << cap.get(cv::CAP_PROP_FRAME_COUNT)
            << "\n";

    cout << "Duration: "
            << cap.get(cv::CAP_PROP_FRAME_COUNT) /
                cap.get(cv::CAP_PROP_FPS)
            << " seconds\n";

    cap.release();

    // Allocate the memory for the target.

    // Release host memory

    return 0;
}
