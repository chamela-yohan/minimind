#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
#include "../network/network.h"
#include "loss.h"

class Trainer {
public:

    Network& net;
    double learning_rate;


    Trainer(Network& net, double learning_rate)
        : net(net), learning_rate(learning_rate) {}


    // Train on one sample, return its loss
    double train_sample(const Matrix& input, const Matrix& target) {

        // Forward pass — run input through every layer
        Matrix prediction = net.forward(input);

        // Measure loss
        double loss = Loss::mse(prediction, target);

        // Start backprop — initial gradient comes from the loss function
        Matrix gradient = Loss::mse_derivative(prediction, target);

        // Flow gradient backwards through every layer
        for (int i = (int)net.layers.size() - 1; i >= 0; i--)
            gradient = net.layers[i].backward(gradient, learning_rate);

        return loss;
    }


    // Train for multiple epochs over all samples
    void train(const std::vector<Matrix>& inputs,
               const std::vector<Matrix>& targets,
               int epochs) {

        std::cout << std::fixed << std::setprecision(6);

        for (int epoch = 0; epoch <= epochs; epoch++) {

            double total_loss = 0.0;
            for (int i = 0; i < (int)inputs.size(); i++)
                total_loss += train_sample(inputs[i], targets[i]);

            double avg_loss = total_loss / inputs.size();

            // Print progress every 500 epochs
            if (epoch % 500 == 0) {
                // Simple ASCII loss bar — width scales with loss
                int bar_len = (int)(avg_loss * 40);
                std::cout << "Epoch " << std::setw(5) << epoch
                          << " | Loss: " << avg_loss << " [";
                for (int b = 0; b < 40; b++)
                    std::cout << (b < bar_len ? "#" : " ");
                std::cout << "]\n";
            }
        }
    }

};