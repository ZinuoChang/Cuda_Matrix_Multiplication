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

// // void matrixVectorMul(const Eigen::MatrixXd& matrix, const Eigen::MatrixXd& vectorMatrix, Eigen::MatrixXd& result) {
// void matrixVectorMul(double* matrix, double* vectorMatrix, double* result, int rows, int cols, int vec_num) {
//     // int rows = matrix.rows();
//     // int cols = matrix.cols();
//     // int vec_num = vectorMatrix.cols();

//     // Allocate device memory
//     double* d_matrix;
//     double* d_vector;
//     double* d_result;
//     cudaMalloc((double**)&d_matrix, rows * cols * sizeof(double));
//     cudaMalloc((double**)&d_vector, cols * vec_num * sizeof(double));
//     cudaMalloc((double**)&d_result, cols * vec_num * sizeof(double));

//     // Copy data to device
//     cudaMemcpy(d_matrix, matrix.data(), rows * cols * sizeof(double), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_vector, vectorMatrix.data(), cols * vec_num * sizeof(double), cudaMemcpyHostToDevice);

//     // Launch kernel
//     dim3 blockSize(1, 1);
//     dim3 threadperblock((vec_num + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
//     matrixVectorMultiplication<<<threadperblock, blockSize>>>(d_matrix, d_vector, d_result, rows, cols, vec_num);

//     // Copy result back to host
//     cudaMemcpy(result.data(), d_result, cols * vec_num * sizeof(double), cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(d_matrix);
//     cudaFree(d_vector);
//     cudaFree(d_result);
// }


// int main(void)
void matrixVectorMul(double* matrix, double* vectorMatrix, double* result, int rows, int cols, int vec_num)
{
    // Define dimensions

    // rows = 1000;
    // cols = 1000;
    // vec_num = 1000;

    // double* matrix = new double[rows * cols];
    // double* vectorMatrix = new double[rows * vec_num];
    // double* result = new double[rows * vec_num];
    double* result_cpu = new double[rows * vec_num];
    
    double *matrix_gpu, *vectorMatrix_gpu, *result_gpu;  

    cudaMalloc(&matrix_gpu, rows * cols * sizeof(double));
    cudaMalloc(&vectorMatrix_gpu, cols * vec_num * sizeof(double));
    cudaMalloc(&result_gpu, cols * vec_num * sizeof(double));




    cudaMemcpy(matrix_gpu, matrix, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vectorMatrix_gpu, vectorMatrix, rows * vec_num * sizeof(double), cudaMemcpyHostToDevice);
    // std::cout << matrix << std::endl;

    // Launch kernel
    dim3 blockSize(4, 4);
    dim3 threadperblock((vec_num + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
    
    
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixVectorMultiplication<<<threadperblock, blockSize>>>(matrix_gpu, vectorMatrix_gpu, result_gpu, rows, cols, vec_num);
    
    cudaDeviceSynchronize();

    cudaMemcpy(result, result_gpu, rows * vec_num * sizeof(double), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time GPU: " << elapsed.count() << " seconds" << std::endl;

    // Free device memory
    cudaFree(matrix);
    cudaFree(vectorMatrix);
    cudaFree(result);

    // auto start1 = std::chrono::high_resolution_clock::now();
    // matrix_multiplication_cpu(matrix, vectorMatrix, result_cpu, rows, cols, vec_num);

    // auto end1 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed1 = end1 - start1;
    // std::cout << "Elapsed time CPU: " << elapsed1.count() << " seconds" << std::endl;

    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < vec_num; j++) {
    //         std::cout << result[i * cols + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < vec_num; j++) {
    //         std::cout << result_cpu[i * cols + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // double error = 0;

    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < vec_num; j++) {
    //         error += abs(result[i * cols + j] - result_cpu[i * cols + j]);
    //     }
    // }
    // std::cout << error << " ";

    // Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(rows, cols);
    // Eigen::MatrixXd vectorMatrix(cols, vec_num);
    // Eigen::MatrixXd result(cols, vec_num);

    // // Randomly generate the vector of VectorXd
    // std::vector<Eigen::VectorXd> vectors(vec_num, Eigen::VectorXd(cols));
    // for (int i = 0; i < vec_num; ++i) {
    //     vectors[i] = Eigen::VectorXd::Random(cols);
    // }

    // // Convert vector of vectors to a matrix
    // for (int i = 0; i < vec_num; ++i) {
    //     vectorMatrix.col(i) = vectors[i];
    // }


    // matrixVectorMul(matrix, vectorMatrix, result);

    // std::cout << "Result:\n" << result << std::endl;


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

    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < vec_num; j++) {
    //         double randomValue = static_cast<double>(rand()) / RAND_MAX;  // Generates a random float between 0 and 1
    //         // std::cout << randomValue << std::endl;
    //         vectorMatrix[i * cols + j] = randomValue;
    //         // std::cout << vectorMatrix[i * cols + j] << std::endl;
    //     }
    // }