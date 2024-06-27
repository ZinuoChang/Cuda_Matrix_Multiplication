#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <chrono>

void matrixVectorMul(double* d_matrix, double* d_vector, double* d_result, int rows, int cols, int vec_num);

inline void matrix_multiplication_cpu(double* d_matrix, double* d_vector, double* d_result, int rows, int cols, int vec_num) {
    for (int i=0; i<rows; i++){
        for (int j=0; j<vec_num; j++){
            double sum = 0.0;
            for (int k=0; k<cols; k++){
                sum += d_matrix[i * cols + k] * d_vector[k * vec_num + j];
            }
            d_result[i * vec_num + j] = sum;
        }
    }
}

#endif // MATRIX_H