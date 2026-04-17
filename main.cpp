#include <iostream>
#include <iomanip>
#include <vector>
#include "src/math/matrix.h"
#include "src/math/activations.h"
#include "src/network/layer.h"
#include "src/network/network.h"
#include "src/training/loss.h"
#include "src/training/trainer.h"
#include "src/io/serializer.h"

// Helper to print XOR predictions for any network
void print_predictions(Network& net) {
    double raw_inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double raw_targets[4]   = {0, 1, 1, 0};

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Input             Expected         Predicted     Verdict\n";
    std::cout << " -------------------------------------------------------------\n";

    for (int i = 0; i < 4; i++) {
        Matrix input(2, 1);
        input.set(0, 0, raw_inputs[i][0]);
        input.set(1, 0, raw_inputs[i][1]);

        double predicted = net.forward(input).get(0, 0);
        std::string verdict = (std::round(predicted) == raw_targets[i]) ? "✓" : "✗";

        std::cout << "  [" << raw_inputs[i][0] << "," << raw_inputs[i][1] << "]"
                  << "     " << raw_targets[i]
                  << "          " << predicted
                  << "      " << verdict << "\n";
    }
}

int main() {
    std::cout << "MiniMind — Save and Load Weights\n\n";

    //  Phase 1: Train and save 
    std::cout << "=== Phase 1: Train and save ===\n\n";

    Network net;
    net.add_layer(2, 4, Activation::SIGMOID);
    net.add_layer(4, 1, Activation::SIGMOID);

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

    Trainer trainer(net, 0.1);
    trainer.train(inputs, targets, 5000);

    std::cout << "\nPredictions before saving:\n";
    print_predictions(net);

    // Save weights to disk
    Serializer::save(net, "weights.bin");


    //  Phase 2: Fresh network, load weights, predict 
    std::cout << "\n=== Phase 2: Fresh network + load weights ===\n\n";

    // Brand new network — random weights, knows nothing
    Network net2;
    net2.add_layer(2, 4, Activation::SIGMOID);
    net2.add_layer(4, 1, Activation::SIGMOID);

    std::cout << "Predictions BEFORE loading (random weights):\n";
    print_predictions(net2);

    // Load trained weights into it
    Serializer::load(net2, "weights.bin");

    std::cout << "\nPredictions AFTER loading (trained weights):\n";
    print_predictions(net2);

    return 0;
}