#include "matrix.h"
#include <Eigen/Dense>

int main (void){
    // Define dimensions
    int rows = 3;
    int cols = 4;
    int vec_num = 5;

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

    // Allocate memory for arrays to store matrix and vector data
    double* matrix_array = new double[matrix.size()];
    double* vectorMatrix_array = new double[vectorMatrix.size()];
    double* result_array = new double[result.size()];

    // Assign the value in eigen matrix into array
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matrix_array, matrix.rows(), matrix.cols()) = matrix;
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(vectorMatrix_array, vectorMatrix.rows(), vectorMatrix.cols()) = vectorMatrix;
    
    // std::cout << "Matrix:" << std::endl << matrix << std::endl;
    // std::cout << std::endl;

    // std::cout << "Array:" << std::endl;
    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < cols; j++) {
    //         std::cout << matrix_array[i * cols + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    matrixVectorMul(matrix_array, vectorMatrix_array, result_array, rows, cols, vec_num);

    // auto start = std::chrono::high_resolution_clock::now();
    
    // result = matrix * vectorMatrix;

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Elapsed time Eigen: " << elapsed.count() << " seconds" << std::endl;

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> result_transform(result_array, rows, vec_num);
    // result = result_transform;
    result = matrix * vectorMatrix;

    // std::cout << "Result:" << std::endl << result << std::endl;
    // std::cout << std::endl;

    // std::cout << "Result1:" << std::endl << result_transform << std::endl;
    // std::cout << std::endl;
    
    return 0;
}