#include "matrix.h"
#include <Eigen/Dense>

int main (void){
    int rows = 1000;
    int cols = 1000;
    int vec_num = 200;

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

    double* matrix1 = new double[rows * cols];
    double* vectorMatrix1 = new double[rows * vec_num];
    double* result1 = new double[rows * vec_num];

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix1[i * cols + j] = matrix(i, j);
        }
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < vec_num; ++j) {
            vectorMatrix1[i * vec_num + j] = vectorMatrix(i, j);
        }
    }


    // std::cout << matrix <<std::endl;

    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < cols; j++) {
    //         std::cout << matrix1[i * cols + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << vectorMatrix <<std::endl;

    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < vec_num; j++) {
    //         std::cout << vectorMatrix1[i * vec_num + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    matrixVectorMul(matrix1, vectorMatrix1, result1, rows, cols, vec_num);

    auto start = std::chrono::high_resolution_clock::now();
    
    result = matrix * vectorMatrix;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time Eigen: " << elapsed.count() << " seconds" << std::endl;

    

    // std::cout << "Eigen" <<std::endl;
    // std::cout << result <<std::endl;

    // std::cout << "GPU" <<std::endl;
    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < vec_num; j++) {
    //         std::cout << result1[i * vec_num + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    return 0;
}