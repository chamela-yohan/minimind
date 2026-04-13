#pragma once
#include <random>
#include <stdexcept>
#include "../math/matrix.h"
#include "../math/activations.h"

// Which activation this layer uses
enum class Activation
{
    SIGMOID,
    RELU
};

class DenseLayer
{
public:
    int input_size;
    int output_size;
    Activation activation;

    Matrix weights; // shape: (output_size x input_size)
    Matrix bias;    // shape: (output_size x 1)

    // Stored during forward pass - needed for backpropagation later
    Matrix last_input; // What came in
    Matrix last_z;     // raw output before activation (weights*inputs + bias)

    DenseLayer(int input_size, int output_size, Activation activation)
        : input_size(input_size),
          output_size(output_size),
          activation(activation),
          weights(output_size, input_size),
          bias(output_size, 1),
          last_input(input_size, 1),
          last_z(output_size, 1)
    {
        init_weights();
    }

    // Forward Pass
    Matrix forward(const Matrix &input)
    {
        if (input.rows != input_size || input.cols != 1)
            throw std::invalid_argument("DenseLayer forward: wrong input shape");

        last_input = input;

        // z = weights * input + bias
        last_z = weights.multiply(input).add(bias);

        // Activation
        if (activation == Activation::SIGMOID)
            return apply_sigmoid(last_z);
        else
            return apply_relu(last_z);
    }

    // Print layer info
    void print_info(const std::string &name = "Layer") const
    {
        std::cout << name << ": "
                  << input_size << " inputs -> "
                  << output_size << " outputs | "
                  << (activation == Activation::SIGMOID ? "sigmoid" : "relu")
                  << "\n";
        weights.print("  Weights");
        bias.print("  Bias");
    }

private:
    // Weight initialization
    void init_weights()
    {
       double range = std::sqrt(1.0 / input_size);

        std::mt19937 rng(42);  // 42 = fixed seed, same result every run
        std::uniform_real_distribution<double> dist(-range, range);

        for (int i = 0; i < weights.rows; i++)
            for (int j = 0; j < weights.cols; j++)
                weights.data[i][j] = dist(rng);

    }
};