#include <iostream>
#include <iomanip>
#include <vector>
#include "src/math/matrix.h"
#include "src/math/activations.h"
#include "src/network/layer.h"
#include "src/network/network.h"
#include "src/training/loss.h"
#include "src/training/trainer.h"

int main() {
    std::cout << "MiniMind — Training on XOR\n\n";

    // Build the network
    Network net;
    net.add_layer(2, 4, Activation::SIGMOID);
    net.add_layer(4, 1, Activation::SIGMOID);
    net.print_info();

    // XOR training data
    std::vector<Matrix> inputs;
    std::vector<Matrix> targets;

    double raw_inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double raw_targets[4]   = {0, 1, 1, 0};

    for (int i = 0; i < 4; i++) {
        Matrix in(2, 1);
        in.set(0, 0, raw_inputs[i][0]);
        in.set(1, 0, raw_inputs[i][1]);
        inputs.push_back(in);

        Matrix tgt(1, 1);
        tgt.set(0, 0, raw_targets[i]);
        targets.push_back(tgt);
    }

    // Train
    std::cout << "\nTraining...\n\n";
    Trainer trainer(net, 0.1);
    trainer.train(inputs, targets, 5000);

    // Final predictions
    std::cout << "\nFinal predictions\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Input             Expected         Predicted     Verdict\n";
    std::cout << " -------------------------------------------------------------\n";

    double total_loss = 0.0;
    for (int i = 0; i < 4; i++) {
        Matrix output = net.forward(inputs[i]);
        double predicted = output.get(0, 0);
        double loss = Loss::mse(output, targets[i]);
        total_loss += loss;

        // Round to nearest integer for a binary verdict
        std::string verdict = (std::round(predicted) == raw_targets[i]) ? "✓" : "✗";

        std::cout << "  [" << raw_inputs[i][0] << "," << raw_inputs[i][1] << "]"
                  << "     " << raw_targets[i]
                  << "          " << predicted
                  << "      " << verdict << "\n";
    }

    std::cout << " -------------------------------------------------------------\n";
    std::cout << "  Final avg loss: " << total_loss / 4 << "\n";

    return 0;
}