#include <iostream>
#include <iomanip>
#include "src/math/matrix.h"
#include "src/math/activations.h"
#include "src/network/layer.h"

using namespace std;

int main()
{
    std::cout << "MiniMind — Activation Functions Test\n\n";
    std::cout << std::fixed << std::setprecision(4);

     // Create a layer: 2 inputs, 3 outputs, sigmoid activation
    DenseLayer layer(2, 3, Activation::SIGMOID);
    layer.print_info("Hidden layer");

    // Create a (2x1) input — column vector
    Matrix input(2, 1);
    input.set(0, 0, 1.0);   // first input
    input.set(1, 0, 0.0);   // second input
    input.print("\nInput [1, 0]");

    // Run the forward pass
    Matrix output = layer.forward(input);
    output.print("\nLayer output (after sigmoid)");

    // Show what was stored internally
    layer.last_z.print("\nStored z (before activation)");
    layer.last_input.print("Stored input");

    // Try a second input
    std::cout << "\n--- Second input [0, 1] ---\n";
    Matrix input2(2, 1);
    input2.set(0, 0, 0.0);
    input2.set(1, 0, 1.0);
    layer.forward(input2).print("Output");
    
    return 0;
}