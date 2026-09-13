#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "img.h"

#define RGB_COMPONENT_COLOR 255

cl_int LoadImg(const char *path, Image* img)
{
    char buff[16];
    FILE *fp;
    int c, rgb_comp_color;
    //open PPM file for reading
    fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Unable to open file '%s'\n", path);
        return CL_INVALID_VALUE;
    }

    //read image format
    if (!fgets(buff, sizeof(buff), fp)) {
        perror(path);
        return CL_INVALID_VALUE;
    }

    //check the image format
    if (buff[0] != 'P' || buff[1] != '6') {
        fprintf(stderr, "Invalid image format (must be 'P6')\n");
        return CL_INVALID_VALUE;
    }

    //check for comments
    c = getc(fp);
    while (c == '#') {
        while (getc(fp) != '\n') ;
        c = getc(fp);
    }

    ungetc(c, fp);
    //read image size information
    if (fscanf(fp, "%d %d", &img->shape[1], &img->shape[0]) != 2) {
        fprintf(stderr, "Invalid image size (error loading '%s')\n", path);
        return CL_INVALID_VALUE;
    }

    //read rgb component
    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
        fprintf(stderr, "Invalid rgb component (error loading '%s')\n", path);
        return CL_INVALID_VALUE;
    }

    //check rgb component depth
    if (rgb_comp_color != RGB_COMPONENT_COLOR) {
        fprintf(stderr, "'%s' does not have 8-bits components\n", path);
        return CL_INVALID_VALUE;
    }

    while (fgetc(fp) != '\n') ;
    //memory allocation for pixel data
    unsigned char* data = (unsigned char *)malloc(img->shape[0] * img->shape[1] * IMAGE_CHANNELS * sizeof(char));
    img->data = (int *)malloc(img->shape[0] * img->shape[1] * IMAGE_CHANNELS * sizeof(int));

    if (!data || !img->data) {
        fprintf(stderr, "Unable to allocate memory\n");
        return CL_INVALID_VALUE;
    }

    //read pixel data from file
    if (fread(data, IMAGE_CHANNELS * img->shape[0], img->shape[1], fp) != img->shape[1]) {
        fprintf(stderr, "Error loading image '%s'\n", path);
        return CL_INVALID_VALUE;
    }

    int count = img->shape[0] * img->shape[1] * 3;
    for (int i = 0; i < count; i++)
    {
        img->data[i] = (int)data[i] / 255;
    }

    fclose(fp);
    free(data);

    return CL_SUCCESS;
}

cl_int LoadImgRaw(const char *path, Image* img)
{
    FILE *data_file;

    data_file = fopen(path, "r");
    if (!data_file) {
        printf("Could not find file.\n");
        return CL_INVALID_VALUE;
    } // Error opening file

    unsigned int rows       = 0;
    unsigned int cols       = 0;
    unsigned int channels   = 0;
    
    if (fscanf(data_file, "# (%u, %u, %u)\n", &rows, &cols, &channels) == EOF) {
        printf("Could not parse header.\n");
        return CL_INVALID_VALUE; // Error parsing dimensions
    }

    if (rows == 0){
        rows = 1;
    }
    if (cols == 0) {
        cols = 1;
    }
    if (channels == 0){
        channels = 3;
    }

    img->shape[0] = rows;
    img->shape[1] = cols;
    img->shape[2] = channels;

    img->data = malloc(sizeof(int) * rows * cols * channels);
    if (!img->data){ // Error mallocing matrix data
        return CL_OUT_OF_HOST_MEMORY;
    }

    int n = 0;
    while (fscanf(data_file, "%d", &(img->data[n++])) != EOF);

    fclose(data_file);

    return CL_SUCCESS;
}

cl_int LoadStride(const char *dir, int *stride) {
    char path[256];
    sprintf(path, "%s/stride.raw", dir);
    FILE *fp = fopen(path, "r");
    if (!fp) return CL_INVALID_VALUE;
    fscanf(fp, "%d", stride);
    fclose(fp);
    return CL_SUCCESS;
}


// Writes the same text format LoadImgRaw reads, so images round-trip and a
// saved result can be compared against a dataset reference directly.
cl_int SaveImg(const char *path, Image* img)
{
    FILE *data_file;

    data_file = fopen(path, "w");
    if (!data_file) // Error opening file
        return CL_INVALID_VALUE;

    unsigned int rows = img->shape[0];
    unsigned int cols = img->shape[1];

    // Callers are not required to populate shape[2], so the channel count
    // comes from IMAGE_CHANNELS rather than the shape.
    if (fprintf(data_file, "# (%u, %u, %u)\n", rows, cols, IMAGE_CHANNELS) < 0)
        return CL_INVALID_VALUE; // Error writing dimensions

    for (unsigned int r = 0; r < rows; r++)
    {
        for (unsigned int i = 0; i < cols * IMAGE_CHANNELS; i++)
        {
            if (fprintf(data_file, "%d ", img->data[cols * IMAGE_CHANNELS * r + i]) < 0)
                return CL_INVALID_VALUE; // Error writing pixel data
        }
        fprintf(data_file, "\n");
    }

    fclose(data_file);

    return CL_SUCCESS;
}

cl_int CheckImg(Image *truth, Image *student)
{
    if (truth->shape[0] != student->shape[0] || truth->shape[1] != student->shape[1])
    {
        printf("!!SOLUTION IS NOT CORRECT!!\n");
        return CL_INVALID_VALUE;
    }

    int count = truth->shape[0] * truth->shape[1] * IMAGE_CHANNELS;
    for (int i = 0; i < count; i++)
    {
        // float epsilon = fabs(truth->data[i]) * 0.1f;
        int diff = truth->data[i] - student->data[i];
        if (diff !=  0)
        {
            printf("!!SOLUTION IS NOT CORRECT!! Expected: %d, Found %df at %d\n", truth->data[i], student->data[i], i);
            return CL_INVALID_VALUE;
        }
    }

    printf("!!SOLUTION IS CORRECT!!\n");
    return CL_SUCCESS;
}
