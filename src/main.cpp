//main.cpp


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

void OpenCLEventTracker(int frame_counter, int width, int height, const vector<int8_t>& event_map_idx)
{
    const int event_map_entries = height * width;
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

// test to see how many pixels are on/off for events

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
    imwrite("event_layer1.png", event_image);
    cout << "Saved middle frame debug image" << endl;
}

}


int main() {

    VideoCapture cap("video/Video.mov"); // input video

    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);
    Size frame_size(frame_width, frame_height);
    int vid_ext = VideoWriter::fourcc('m', 'p', '4', 'v');
    string file_name = "video_out.mp4";
    string file_name2 = "event_out.mp4";

    VideoWriter vid(file_name, vid_ext, fps, frame_size, true); // centroid output
    VideoWriter vid2(file_name2, vid_ext, fps, frame_size, false); // event output

    int frame_counter = 0;


    if (!cap.isOpened()) {
        std::cerr << "Error: could not open video\n";
        return -1;
    }

    if (!cap.read(frame)) {
      std::cerr << "Error: could not read first frame\n";
      return -1;
   }

   cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

   previous_gray = frame_gray.clone();

   int height = frame_gray.rows;
   int width = frame_gray.cols;

   int event_map_entries = height * width;


    // Load external OpenCL kernel code
    char *kernel_source = OclLoadKernel(KERNEL_PATH); // Load kernel source

    // Device input and output buffers
    cl_mem device_current_f, device_previous_f, device_event_map;
    cl_mem device_surface, device_output_surface;

    size_t global_item_size[2];
    cl_int err;

    cl_device_id device_id;    // device ID
    cl_context context;        // context
    cl_command_queue queue;    // command queue
    cl_program program;        // program
    cl_kernel kernel1;          // kernel1
    cl_kernel kernel2;          // kernel2
    cl_kernel kernel3;          // kernel3

    const int threshold = 35;
    const float alpha = .85;
    const float noise_threshold = 40;


    // Platforms and devices
    OclPlatformProp *platforms = NULL;
    cl_uint num_platforms;

    err = OclFindPlatforms((const OclPlatformProp **)&platforms, &num_platforms);
    CHECK_ERR(err, "OclFindPlatforms");

    // ID for the specified kind of device type.
    err = OclGetDeviceWithFallback(&device_id, OCL_DEVICE_TYPE);
    CHECK_ERR(err, "OclGetDeviceWithFallback");

    // Context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    CHECK_ERR(err, "clCreateContext");

    // Command queue
# if __APPLE__
    queue = clCreateCommandQueue(context, device_id, 0, &err);
#else
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
#endif
    CHECK_ERR(err, "clCreateCommandQueueWithProperties");

    // Create program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, NULL, &err);
    CHECK_ERR(err, "clCreateProgramWithSource");

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    CHECK_ERR(err, "clBuildProgram");

    // Computer kernels for the 3 layers
    kernel1 = clCreateKernel(program, "generate_events", &err);
    CHECK_ERR(err, "clCreateKernel");
    kernel2 = clCreateKernel(program, "generate_sae", &err);
    CHECK_ERR(err, "clCreateKernel");
    kernel3 = clCreateKernel(program, "generate_boxFilter", &err);
    CHECK_ERR(err, "clCreateKernel");

    // Allocate GPU memory
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
    vector<float> initial_surface(size_d, 0.0f);
    err = clEnqueueWriteBuffer(queue, device_surface, CL_TRUE, 0, size_d * sizeof(float), initial_surface.data(),0, NULL,NULL);
    CHECK_ERR(err, "Initialize device_surface");
    device_output_surface = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_e * sizeof(float), NULL, &err );
    CHECK_ERR(err, "clCreateBuffer device_output_surface");
    // define global work sizes
    global_item_size[0] = height;
    global_item_size[1] = width;

    vector<int8_t> event_map_idx(event_map_entries);

    while (cap.read(frame)) {
    std::cout << "Read frame successfully\n";

    cvtColor(frame,frame_gray, COLOR_BGR2GRAY);

    //Copy memory to the GPU here
    err = clEnqueueWriteBuffer(queue, device_current_f, CL_TRUE, 0, size_a* sizeof(uchar), frame_gray.data, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer device_current_f");
    err = clEnqueueWriteBuffer(queue, device_previous_f, CL_TRUE, 0, size_b* sizeof(uchar), previous_gray.data, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer device_previous_f");

    frame_counter++;

    // Layer 1 arguments
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

    //Launch the GPU Kernel
    err = clEnqueueNDRangeKernel(queue, kernel1, 2, NULL, global_item_size, NULL, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueNDRangeKernel");
    //Copy the GPU memory back to the CPU here
    err = clEnqueueReadBuffer(queue, device_event_map, CL_TRUE, 0, size_c* sizeof(char), event_map_idx.data(), 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueReadBuffer");
    OpenCLEventTracker(frame_counter,width,height,event_map_idx);

     // Layer 2 arguments
    err = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &device_event_map);
    CHECK_ERR(err, "clSetKernelArg 0");
    err |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), &device_surface);
    CHECK_ERR(err, "clSetKernelArg 1");
    err |= clSetKernelArg(kernel2, 2, sizeof(float), &alpha);
    CHECK_ERR(err, "clSetKernelArg 2");
    err |= clSetKernelArg(kernel2, 3, sizeof(const int), &width);
    CHECK_ERR(err, "clSetKernelArg 3");
    err |= clSetKernelArg(kernel2, 4, sizeof(const int), &height);
    CHECK_ERR(err, "clSetKernelArg 4");

    //Launch the GPU Kernel here
    err = clEnqueueNDRangeKernel(queue, kernel2, 2, NULL, global_item_size, NULL, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueNDRangeKernel");
    //Copy the GPU memory back to the CPU here
    err = clEnqueueReadBuffer(queue, device_surface, CL_TRUE, 0, size_d* sizeof(float), initial_surface.data(), 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueReadBuffer");

    // create layer 2 png
    if (frame_counter == 20)
{
    Mat surface_image2(height, width, CV_8UC1);

    for (int i = 0; i < event_map_entries; i++)
    {
        float value = initial_surface[i];

        if (value < 0)
            value = 0;

        if (value > 255)
            value = 255;

        surface_image2.data[i] = (uchar)value;
    }

    imwrite("event_layer2.png", surface_image2);
    cout << "Saved Layer 2 surface image" << endl;
}

     //Layer 3 arguments
    err = clSetKernelArg(kernel3, 0, sizeof(cl_mem), &device_surface);
    CHECK_ERR(err, "clSetKernelArg 0");
    err |= clSetKernelArg(kernel3, 1, sizeof(cl_mem), &device_output_surface);
    CHECK_ERR(err, "clSetKernelArg 1");
    err |= clSetKernelArg(kernel3, 2, sizeof(float), &noise_threshold);
    CHECK_ERR(err, "clSetKernelArg 2");
    err |= clSetKernelArg(kernel3, 3, sizeof(const int), &width);
    CHECK_ERR(err, "clSetKernelArg 3");
    err |= clSetKernelArg(kernel3, 4, sizeof(const int), &height);
    CHECK_ERR(err, "clSetKernelArg 4");

    //Launch the GPU Kernel here
    err = clEnqueueNDRangeKernel(queue, kernel3, 2, NULL, global_item_size, NULL, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueNDRangeKernel");
    //Copy the GPU memory back to the CPU here
    err = clEnqueueReadBuffer(queue, device_output_surface, CL_TRUE, 0, size_e* sizeof(float), initial_surface.data(), 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueReadBuffer");

    Mat surface_image(height, width, CV_8UC1);
    // create surface image for layer 3 and used to write for event_video
    for (int i = 0; i < event_map_entries; i++)
    {
        float value = initial_surface[i];

        if (value < 0)
            value = 0;

        if (value > 255)
            value = 255;

        surface_image.data[i] = (uchar)value;
    }


    if (frame_counter == 20)
{
    imwrite("event_layer3.png", surface_image);
    cout << "Saved Layer 3 surface image" << endl;
}
// Centroid calculation
    double weighted_x = 0.0;
    double weighted_y = 0.0;
    double total_weight = 0.0;
    
for (int row = 0; row < height; row++)
{
    for (int col = 0; col < width; col++)
    {
        int idx = row * width + col;
        float intensity = initial_surface[idx];
        if (row < height * 0.95 && col > width * 0.05 && intensity > 20){

        weighted_x += col * intensity;
        weighted_y += row * intensity;
        total_weight += intensity;
        }
    }
}
if(total_weight > 0)
{
    float cx = weighted_x/total_weight;
    float cy = weighted_y/total_weight;

    circle(frame,Point(cx, cy),50,Scalar(0, 0, 255), 2);

    cout << "Centroid: (" << cx << ", " << cy << ")" << endl; //print centroid values

    // centroid png
    if(frame_counter == 20 && total_weight > 0)
{
    imwrite("centroid_debug.png", frame);
}
}
    vid.write(frame);
    vid2.write(surface_image);
    previous_gray = frame_gray.clone(); // make sure previous frame is set after processing
}
// test to see fps, frame count, duration

    cout << "FPS: "
          << fps
          << "\n";

    cout << "Frame count: "
            << cap.get(cv::CAP_PROP_FRAME_COUNT)
            << "\n";

    cout << "Duration: "
            << cap.get(cv::CAP_PROP_FRAME_COUNT) /
                cap.get(cv::CAP_PROP_FPS)
            << " seconds\n";
    vid.release();
    vid2.release();
    cap.release();

    // Release host memory
    clReleaseMemObject(device_current_f);
    clReleaseMemObject(device_previous_f);
    clReleaseMemObject(device_event_map);
    clReleaseMemObject(device_surface);
    clReleaseMemObject(device_output_surface);

    clReleaseKernel(kernel1);
    clReleaseKernel(kernel2);
    clReleaseKernel(kernel3);

    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
