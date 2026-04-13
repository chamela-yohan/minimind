#pragma once
#include <vector>
#include <iostream>
#include <stdexcept>

class Matrix
{
public:
    int rows;
    int cols;

    std::vector<std::vector<double>> data;

    Matrix(int rows, int cols, double init = 0.0)
        : rows(rows),
          cols(cols),
          data(rows, std::vector<double>(cols, init)) {}

    // Get a vlaue
    double get(int row, int col) const
    {
        if (row < 0 || row >= rows || col < 0 || col >= cols)
            throw std::out_of_range("Matrix index out of bounds");
        return data[row][col];
    }

    // Set a value
    void set(int row, int col, double value)
    {
        if (row < 0 || row >= rows || col < 0 || col >= cols)
            throw std::out_of_range("Matrix index out of bounds");
        data[row][col] = value;
    }

    // Print to terminal
    void print(const std::string &label = "") const
    {
        if (!label.empty())
            std::cout << label << ":\n";

        for (int i = 0; i < rows; i++)
        {
            std::cout << " [ ";
            for (int j = 0; j < cols; j++)
            {
                std::cout << data[i][j];
                if (j < cols - 1)
                    std::cout << ", ";
            }
            std::cout << "]\n";
        }
    }

    // Add two matrix
    Matrix add(const Matrix &other)
    {
        if (rows != other.rows || cols != other.cols)
            throw std::invalid_argument("Matrix add: size mismatch");

        Matrix result(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result.data[i][j] = result.data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    // Multiply two matrices (dot product)
    Matrix multiply(const Matrix &other)
    {
        if (cols != other.rows)
            throw std::invalid_argument("Matrix multiply: incompatible dimensions");

        Matrix result(rows, other.cols);

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < other.cols; j++)
                for (int k = 0; k < cols; k++)
                    result.data[i][j] += data[i][k] * other.data[k][j];

        return result;
    }

    // Transpose
    Matrix transpose() const
    {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.data[j][i] = data[i][j];

        return result;
    }

    // Scale (Multiply every element by a single number)
    Matrix scale(double factor) const{
        Matrix result(rows,cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result.data[i][j] = data[i][j] * factor;

        return result;
    }
};
