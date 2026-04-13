#pragma once
#include <cmath>
#include "matrix.h"

// Sigmoid
inline double sigmoid(double x)
{
    return 1 / (1 + std::exp(-x));
}

// Derivative of sigmoid - for backprop
inline double sigmoid_derivative(double sigmoid_output)
{
    return sigmoid_output * (1.0 - sigmoid_output);
}

// Relu
inline double relu(double x)
{
    return x > 0.0 ? x : 0.0;
}

// Derivative of Relu - 1 if input was positive, 0 otherwise
inline double relu_derivative(double x)
{
    return x > 0.0 ? 1.0 : 0.0;
}

// Apply to whole matrix

inline Matrix apply_sigmoid(const Matrix &m)
{
    Matrix result(m.rows, m.cols);

    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++)
            result.data[i][j] = sigmoid(m.data[i][j]);

    return result;
}

inline Matrix apply_sigmoid_derivative(const Matrix& m) {
    Matrix result(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++)
            result.data[i][j] = sigmoid_derivative(m.data[i][j]);
    return result;
}

inline Matrix apply_relu(const Matrix& m) {
    Matrix result(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++)
            result.data[i][j] = relu(m.data[i][j]);
    return result;
}

inline Matrix apply_relu_derivative(const Matrix& m) {
    Matrix result(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++)
            result.data[i][j] = relu_derivative(m.data[i][j]);
    return result;
}
