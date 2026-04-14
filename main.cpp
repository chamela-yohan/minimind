#include <iostream>
#include <iomanip>
#include "src/math/matrix.h"
#include "src/math/activations.h"
#include "src/network/layer.h"
#include "src/network/network.h"

using namespace std;

int main()
{
    std::cout << "MiniMind — Activation Functions Test\n\n";
    std::cout << std::fixed << std::setprecision(4);

     // Build the XOR network
    Network net;
    net.add_layer(2, 4, Activation::RELU);      // hidden layer
    net.add_layer(4, 1, Activation::SIGMOID);   // output layer
    net.print_info();

    // The four XOR inputs
    // XOR truth table:
    //   [0,0] -> 0
    //   [0,1] -> 1
    //   [1,0] -> 1
    //   [1,1] -> 0
    double inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double targets[4]   = {0, 1, 1, 0};

    std::cout << "\nForward pass (untrained — random predictions):\n";
    std::cout << "  Input             Expected           Predicted\n";
    std::cout << "  ------------------------------------------------\n";

    for (int i = 0; i < 4; i++) {
        // Build input column vector (2x1)
        Matrix input(2, 1);
        input.set(0, 0, inputs[i][0]);
        input.set(1, 0, inputs[i][1]);

        // Run through network
        Matrix output = net.forward(input);
        double predicted = output.get(0, 0);

        std::cout << "  [" << inputs[i][0] << ", " << inputs[i][1] << "]"
                  << "    " << targets[i]
                  << "          " << predicted << "\n";
    }

    std::cout << "\nPredictions are random — network is untrained.\n";
    std::cout << "After training (Step 8) these should approach: 0, 1, 1, 0\n";
    
    return 0;
}