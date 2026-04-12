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
    double get(int row, int col) const {
        if (row < 0 || row >= rows || col < 0 || col >= cols)
            throw std::out_of_range("Matrix index out of bounds");
        return data[row][col];
    }

    // Set a value
    void set(int row, int col, double value){
        if (row < 0 || row >= rows || col < 0 || col >= cols)
            throw std::out_of_range("Matrix index out of bounds");
        data[row][col] = value;
    }
    
    // Print to terminal
    void print(const std::string& label = "") const {
        if(!label.empty())
            std::cout << label << ":\n";
        
        for (int i = 0; i < rows; i++){
            std::cout << " [ ";
            for (int j = 0; j < cols; j++){
                std::cout << data[i][j];
                if (j < cols - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }
        
    }

};
