#pragma once
#include "../math/matrix.h"

class Loss
{
public:
    // Mean squared error
    // Both matrices must be the same shape
    static double mse(const Matrix &predicted, const Matrix &target)
    {
        if (predicted.rows != target.rows || predicted.cols != target.cols)
            throw std::invalid_argument("Loss MSE: shape mismatch");

        double total = 0.0;
        int n = predicted.rows * predicted.cols;

        for (int i = 0; i < predicted.rows; i++)
            for (int j = 0; j < predicted.cols; j++)
            {
                double diff = predicted.data[i][j] - target.data[i][j];
                total += diff * diff;
            }

        return total / n;
    }

    // MSE Derivative
    // Returns a matrix of the same shape as predicted
    // Each element: (2/n) * (predicted - target)
    // Single backprop uses to correct the output layer
    static Matrix mse_derivative(const Matrix& predicted, const Matrix& target){
        if (predicted.rows != target.rows || predicted.cols != target.cols)
            throw std::invalid_argument("Loss MSE derivative: shape mismatch");

        int n = predicted.rows * predicted.cols;
        Matrix result(predicted.rows, predicted.cols);

        for (int i = 0; i < predicted.rows; i++)
            for (int j = 0; j < predicted.cols; j++)
                result.data[i][j] = (2.0 / n) * (predicted.data[i][j] - target.data[i][j]);

        return result;
    }
};