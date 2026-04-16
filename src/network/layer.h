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

    // --- Backward pass ---
    // Receives the gradient flowing in from the layer ahead of us
    // Returns the gradient to pass back to the layer behind us
    Matrix backward(const Matrix &output_gradient, double learning_rate)
    {

        // Step 1: compute activation derivative at the stored raw output
       Matrix activation_grad(output_size, 1);
        if (activation == Activation::SIGMOID) {
            // sigmoid_derivative expects the ACTIVATED output, not raw z
            // so we apply sigmoid to last_z first, then compute derivative
            Matrix sig_output = apply_sigmoid(last_z);
            activation_grad = apply_sigmoid_derivative(sig_output);
        } else {
            // relu_derivative just checks sign of z — raw value is correct here
            activation_grad = apply_relu_derivative(last_z);
        }

        // Step 2: delta — how much did this layer's raw output affect the loss?
        // Hadamard product: combine incoming error with local activation slope
        Matrix delta = output_gradient.element_multiply(activation_grad);

        // Step 3: compute gradients for weights and biases
        Matrix weight_grad = delta.multiply(last_input.transpose());
        Matrix bias_grad = delta;

        // Step 4: compute gradient to pass back to the previous layer
        Matrix input_gradient = weights.transpose().multiply(delta);

        // Step 5: update weights and biases (gradient descent)
        weights = weights.add(weight_grad.scale(-learning_rate));
        bias = bias.add(bias_grad.scale(-learning_rate));

        return input_gradient;
    }

private:
    // Weight initialization
private:
    void init_weights()
    {
        // Static counter — increments each time a layer is created
        // Ensures every layer gets unique random weights
        static int seed_counter = 0;
        std::mt19937 rng(42 + seed_counter++);
        std::uniform_real_distribution<double> dist(
            -std::sqrt(1.0 / input_size),
            std::sqrt(1.0 / input_size));

        for (int i = 0; i < weights.rows; i++)
            for (int j = 0; j < weights.cols; j++)
                weights.data[i][j] = dist(rng);
    }
};