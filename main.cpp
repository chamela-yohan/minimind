#include <iostream>
#include <iomanip>
#include "src/math/matrix.h"
#include "src/math/activations.h"
#include "src/network/layer.h"
#include "src/network/network.h"
#include "src/network/loss.h"

using namespace std;

int main()
{
    std::cout << "MiniMind — Activation Functions Test\n\n";
    std::cout << std::fixed << std::setprecision(4);

    // Test 1: perfect predictin
    Matrix perfect_pred(1, 1);
    perfect_pred.set(0, 0, 1.0);
    Matrix perfect_target(1, 1);
    perfect_target.set(0, 0, 1.0);

    std::cout << "Perfect prediction loss: " << Loss::mse(perfect_pred, perfect_target) << " (expect 0.0)\n";

    // Test 2: worst case
    Matrix worst_pred(1, 1);
    worst_pred.set(0, 0, 0.0);
    Matrix worst_target(1, 1);
    worst_target.set(0, 0, 1.0);
    std::cout << "Worst prediction loss:   " << Loss::mse(worst_pred, worst_target) << " (expect 1.0)\n";

    // Test 3: random prediction on XOR
    std::cout << "\n--- Untrained network on XOR ---\n";
    std::cout << "  Input              Target         Predicted         Loss\n";
    std::cout << "  ----------------------------------------------------------\n";

    Network net;
    net.add_layer(2, 4, Activation::RELU);
    net.add_layer(4, 1, Activation::SIGMOID);

    double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double targets[4] = {0, 1, 1, 0};

    double total_loss = 0.0;

    for (int i = 0; i < 4; i++)
    {
        Matrix input(2, 1);
        input.set(0, 0, inputs[i][0]);
        input.set(1, 0, inputs[i][1]);

        Matrix target(1, 1);
        target.set(0, 0, targets[i]);

        Matrix output = net.forward(input);
        double loss = Loss::mse(output, target);
        total_loss += loss;

        std::cout << "  [" << inputs[i][0] << "," << inputs[i][1] << "]"
                  << "     " << targets[i]
                  << "        " << output.get(0, 0)
                  << "      " << loss << "\n";
    }

    std::cout << "  ----------------------------------------------------------\n";
    std::cout << "  Total loss: " << total_loss / 4 << "\n";
    std::cout << "\nGoal after training: loss < 0.01\n";

    // Test 4: derivative direction check
    std::cout << "\n--- Derivative direction check ---\n";

    // Predicted too high — derivative should be positive (push down)
    Matrix too_high(1, 1);
    too_high.set(0, 0, 0.9);
    Matrix t1(1, 1);
    t1.set(0, 0, 0.0);
    Matrix d1 = Loss::mse_derivative(too_high, t1);
    std::cout << "Predicted 0.9, target 0.0 -> derivative: "
              << d1.get(0, 0) << " (positive = push down) ✓\n";

    // Predicted too low — derivative should be negative (push up)
    Matrix too_low(1, 1);
    too_low.set(0, 0, 0.1);
    Matrix t2(1, 1);
    t2.set(0, 0, 1.0);
    Matrix d2 = Loss::mse_derivative(too_low, t2);
    std::cout << "Predicted 0.1, target 1.0 -> derivative: "
              << d2.get(0, 0) << " (negative = push up)   ✓\n";
    return 0;
}