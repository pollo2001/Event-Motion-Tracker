//main.c from assignment 3

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>
#include "matrix.h"

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

void NeonBlockMatrixMultiply(Matrix *input0, Matrix *input1, Matrix *result)
{
    //@@ Insert code to implement block matrix multiply with ARM Neon intrinsics here
       //define lock size B cache size, BxB sub-matrixß
    int B  = 32;
    int M = input0->shape[0];
    int K = input0->shape[1]; //MxN marix
    int N = input1->shape[1]; //colums of matrix 1

    memset(result->data, 0, M * N * sizeof(int)); //init and zero all of the matrix

   //OUTER LOOPS: block by block
    for (int ii = 0; ii < M; ii += B) { //step by block
        int i_max = (ii + B < M) ? (ii + B) : M; //if not square
        for (int jj = 0; jj < N; jj += B) { 
            int j_max = (jj + B < N) ? (jj + B) : N;
            for (int kk = 0; kk < K; kk +=B) {
                int k_max = (kk + B < K) ? (kk + B) : K;

                //INNER COMPUTATION 
                for (int i = ii; i < i_max; i++) {
                    for (int k = kk; k < k_max; k++) {
                    
                        int32_t a_val = input0->data[i * K + k];
                        //forward scalar A[i][k] to all 4 lanes
                        int32x4_t a_vec = vdupq_n_s32(a_val);

                        //vectorized loop along columns j stepping by 4
                        int j = jj;
                        for (; j < j_max - 4; j += 4) {
                            //load 4 elements of Matrix B
                            int32x4_t b_vec = vld1q_s32(&input1->data[k * N + j]);
                            //load current accumulator values from Matrix C
                            int32x4_t c_vec = vld1q_s32(&result->data[i * N + j]);

                            //multiply and accumulate: c_vec += (a_vec * b_vec), in one go
                            c_vec = vmlaq_s32(c_vec, a_vec, b_vec);

                            //store 4 updated results back into Matrix C
                            vst1q_s32(&result->data[i * N + j], c_vec);
                        }
                        // cleanup
                        for (; j < j_max; j++) {
                            result->data[i * N + j] += a_val * input1->data[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char *argv[]){
    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s <input_file_0> <input_file_1> <answer_file> <output_file>\n", argv[0]);
        return -1;
    }

    const char *input_file_a = argv[1];
    const char *input_file_b = argv[2];
    const char *input_file_c = argv[3];
    const char *input_file_d = argv[4];

    // Host input and output vectors and sizes
    Matrix host_a, host_b, host_c, answer;
    
    cl_int err;

    err = LoadMatrix(input_file_a, &host_a);
    CHECK_ERR(err, "LoadMatrix");

    err = LoadMatrix(input_file_b, &host_b);
    CHECK_ERR(err, "LoadMatrix");

    err = LoadMatrix(input_file_c, &answer);
    CHECK_ERR(err, "LoadMatrix");

    //@@ Update these values for the output rows and cols of the output
    int rows = host_a.shape[0];
    int cols = host_b.shape[1];
    //@@ Do not use the results from the answer matrix

    // Allocate the memory for the target.
    host_c.shape[0] = rows;
    host_c.shape[1] = cols;
    host_c.data = (int *)malloc(sizeof(int) * host_c.shape[0] * host_c.shape[1]);

    // Call your matrix multiply.
    NeonBlockMatrixMultiply(&host_a, &host_b, &host_c);

    // // Call to print the matrix
    // PrintMatrix(&host_c);

    // Save the matrix
    SaveMatrix(input_file_d, &host_c);

    // Check the result of the matrix multiply
    err = CheckMatrix(&answer, &host_c);
    CHECK_ERR(err, "CheckMatrix");

    // Release host memory
    free(host_a.data);
    free(host_b.data);
    free(host_c.data);
    free(answer.data);

    return 0;
}
