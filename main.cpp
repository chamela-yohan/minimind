#include <iostream>
#include <iomanip>
#include "src/math/matrix.h"
#include "src/math/activations.h"

using namespace std;

int main()
{
    std::cout << "MiniMind — Activation Functions Test\n\n";
    std::cout << std::fixed << std::setprecision(4);

    // Test scalar functions
    std::cout << "Sigmoid on scalars:\n";
    std::cout << "  sigmoid(-5) = " << sigmoid(-5) << "\n";
    std::cout << "  sigmoid( 0) = " << sigmoid(0) << "\n";
    std::cout << "  sigmoid( 5) = " << sigmoid(5) << "\n";

    std::cout << "\nReLU on scalars:\n";
    std::cout << "  relu(-3) = " << relu(-3) << "\n";
    std::cout << "  relu( 0) = " << relu(0) << "\n";
    std::cout << "  relu( 4) = " << relu(4) << "\n";

    // Test on a matrix
    // Like a raw output of a layer before activation
    Matrix raw(2, 3);
    raw.set(0, 0, -2.0);
    raw.set(0, 1, 0.5);
    raw.set(0, 2, 1.5);
    raw.set(1, 0, 3.0);
    raw.set(1, 1, -0.5);
    raw.set(1, 2, -1.5);

    raw.print("\nRaw layer output");
    apply_sigmoid(raw).print("\nAfter sigmoid");
    apply_relu(raw).print("\nAfter ReLU");

    // Derivative check
    // sigmoid'(0.5) should be 0.5 * (1 - 0.5) = 0.25
    std::cout << "\nDerivative check:\n";
    std::cout << "  sigmoid_derivative(0.5) = "
              << sigmoid_derivative(0.5) << " (expect 0.25)\n";

    return 0;
}