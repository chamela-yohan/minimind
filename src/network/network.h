#pragma once
#include <vector>
#include "layer.h"

class Network
{

public:
    // All layers stored in order - input -> hidden -> output
    std::vector<DenseLayer> layers;

    // Add la layer to the network
    void add_layer(int input_size, int output_size, Activation activation)
    {
        layers.emplace_back(input_size, output_size, activation);
    }

    // Forward prop
    // Output of each layer feeds directly into the next
    Matrix forward(const Matrix &input)
    {
        Matrix current = input;
        for (auto &layer : layers)
            current = layer.forward(current);

        return current;
    }

    // Print architecture summary
    void print_info() const {
        std::cout << "Network architecture:\n";
        std::cout << "  Layers: " << layers.size() << "\n";
        for (int i = 0; i < (int)layers.size(); i++) {
            std::cout << "  [" << i << "] "
                      << layers[i].input_size << " -> "
                      << layers[i].output_size << " | "
                      << (layers[i].activation == Activation::SIGMOID ? "sigmoid" : "relu")
                      << "\n";
        }
    }

};