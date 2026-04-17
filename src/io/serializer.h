#pragma once
#include <fstream>
#include <string>
#include <stdexcept>
#include "../network/network.h"

class Serializer {
public:

    // Save weights to binary file
    static void save(const Network& net, const std::string& filepath) {
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("Serializer: cannot open file for writing: " + filepath);

        // Write number of layers
        int num_layers = (int)net.layers.size();
        file.write(reinterpret_cast<const char*>(&num_layers), sizeof(int));

        for (const auto& layer : net.layers) {

            // Write layer dimensions
            file.write(reinterpret_cast<const char*>(&layer.input_size),  sizeof(int));
            file.write(reinterpret_cast<const char*>(&layer.output_size), sizeof(int));

            // Write every weight
            for (int i = 0; i < layer.weights.rows; i++)
                for (int j = 0; j < layer.weights.cols; j++) {
                    double val = layer.weights.data[i][j];
                    file.write(reinterpret_cast<const char*>(&val), sizeof(double));
                }

            // Write every bias
            for (int i = 0; i < layer.bias.rows; i++) {
                double val = layer.bias.data[i][0];
                file.write(reinterpret_cast<const char*>(&val), sizeof(double));
            }
        }

        file.close();
        std::cout << "Weights saved to: " << filepath << "\n";
    }


    // Load weights from binary file
    static void load(Network& net, const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("Serializer: cannot open file for reading: " + filepath);

        // Read and verify number of layers
        int num_layers;
        file.read(reinterpret_cast<char*>(&num_layers), sizeof(int));
        if (num_layers != (int)net.layers.size())
            throw std::runtime_error("Serializer: layer count mismatch in file");

        for (auto& layer : net.layers) {

            // Read and verify dimensions
            int input_size, output_size;
            file.read(reinterpret_cast<char*>(&input_size),  sizeof(int));
            file.read(reinterpret_cast<char*>(&output_size), sizeof(int));

            if (input_size != layer.input_size || output_size != layer.output_size)
                throw std::runtime_error("Serializer: layer shape mismatch in file");

            // Read weights
            for (int i = 0; i < layer.weights.rows; i++)
                for (int j = 0; j < layer.weights.cols; j++) {
                    double val;
                    file.read(reinterpret_cast<char*>(&val), sizeof(double));
                    layer.weights.data[i][j] = val;
                }

            // Read biases
            for (int i = 0; i < layer.bias.rows; i++) {
                double val;
                file.read(reinterpret_cast<char*>(&val), sizeof(double));
                layer.bias.data[i][0] = val;
            }
        }

        file.close();
        std::cout << "Weights loaded from: " << filepath << "\n";
    }

};