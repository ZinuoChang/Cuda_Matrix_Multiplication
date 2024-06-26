#include "matrix.h"
#include <Eigen/Dense>

int main (void){
    // Define dimensions
    int rows = 1000;
    int cols = 1000;
    int vec_num = 200;

    // Generate the matrices randomly
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(rows, cols);
    Eigen::MatrixXd vectorMatrix(cols, vec_num);
    Eigen::MatrixXd result(rows, vec_num);

    // Randomly generate the vector of VectorXd
    std::vector<Eigen::VectorXd> vectors(vec_num, Eigen::VectorXd(cols));
    for (int i = 0; i < vec_num; ++i) {
        vectors[i] = Eigen::VectorXd::Random(cols);
    }

    // Convert vector of vectors to a matrix
    for (int i = 0; i < vec_num; ++i) {
        vectorMatrix.col(i) = vectors[i];
    }

    double* matrix_array = new double[rows * cols];
    double* vectorMatrix_array = new double[rows * vec_num];
    double* result_array = new double[rows * vec_num];

    // Assign the value in eigen matrix into array
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix_array[i * cols + j] = matrix(i, j);
        }
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < vec_num; ++j) {
            vectorMatrix_array[i * vec_num + j] = vectorMatrix(i, j);
        }
    }

    matrixVectorMul(matrix_array, vectorMatrix_array, result_array, rows, cols, vec_num);

    auto start = std::chrono::high_resolution_clock::now();
    
    result = matrix * vectorMatrix;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time Eigen: " << elapsed.count() << " seconds" << std::endl;

    
    // Compare the results
    // std::cout << "Eigen" <<std::endl;
    // std::cout << result <<std::endl;

    // std::cout << "GPU" <<std::endl;
    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < vec_num; j++) {
    //         std::cout << result_array[i * vec_num + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    return 0;
}