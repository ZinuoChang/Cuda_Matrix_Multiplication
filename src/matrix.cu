#include <cuda_runtime.h>
#include "matrix.h"
  
// CUDA kernel for matrix-vector multiplication
__global__ void matrixVectorMultiplication(double* d_matrix, double* d_vector, double* d_result, int rows, int cols, int vec_num) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < vec_num) {
        double sum = 0.0;
        for (int i = 0; i < cols; ++i) {
            sum += d_matrix[row * cols + i] * d_vector[i * vec_num + col];
        }
        d_result[row * vec_num + col] = sum;
    }
} 


void matrixVectorMul(double* matrix, double* vectorMatrix, double* result, int rows, int cols, int vec_num)
{   
    double *matrix_gpu, *vectorMatrix_gpu, *result_gpu;  

    cudaMalloc(&matrix_gpu, rows * cols * sizeof(double));
    cudaMalloc(&vectorMatrix_gpu, cols * vec_num * sizeof(double));
    cudaMalloc(&result_gpu, cols * vec_num * sizeof(double));

    // Copy the data from host to device
    cudaMemcpy(matrix_gpu, matrix, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vectorMatrix_gpu, vectorMatrix, rows * vec_num * sizeof(double), cudaMemcpyHostToDevice);


    // Launch kernel
    dim3 blockSize(4, 4);
    dim3 threadperblock((vec_num + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
    
        
    matrixVectorMultiplication<<<threadperblock, blockSize>>>(matrix_gpu, vectorMatrix_gpu, result_gpu, rows, cols, vec_num);
    cudaDeviceSynchronize();
    cudaMemcpy(result, result_gpu, rows * vec_num * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(matrix);
    cudaFree(vectorMatrix);
    cudaFree(result);

}


    // // Assign random floating-point values to each element of the matrix
    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < cols; j++) {
    //         double randomValue = static_cast<double>(rand()) / RAND_MAX;  // Generates a random float between 0 and 1
    //         // std::cout << randomValue << std::endl;
    //         matrix[i * cols + j] = randomValue;
    //         // std::cout << matrix[i * cols + j] << std::endl;
    //     }
    // }