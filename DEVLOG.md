# MiniMind - Dev Log

## Step 1 - Project setup
Set up the project folder stucture and verified the build works on Windows using g++
Starting point: a clean C++ executable that compiles and runs.

## Step 2 - Matrix class
Build the core Matrix data structure using std::vector<vector<double>>.
Supports construction, get/set with bounds checking, and terminal printing.
No math operations yet - data structure first, operations next.

## Step 3 — Matrix math operations
Added add, multiply, transpose, and scale to the Matrix class.
All operations return new matrices (non-destructive) and validate dimensions.
Matrix multiply was the key one — implemented the three-loop dot product.

## Step 4 — Activation functions
Implemented sigmoid and ReLU (and their derivatives) in activations.h.
Activation functions are what give neural networks the ability to learn
non-linear patterns — without them, stacking layers has no effect.
Derivatives are needed for backpropagation in a later step.

## Step 5 — Dense layer
Built the DenseLayer class — the core building block of the network.
Each layer holds a weight matrix and bias vector, initialized with Xavier initialization.
Forward pass computes z = weights * input + bias, then applies the activation.
Stores last_input and last_z during the forward pass — will be needed for backprop.

## Step 6 — Network class
Built the Network class that chains multiple DenseLayer objects together.
forward() passes data through every layer in sequence — output of one
becomes input of the next. Wired up the full XOR architecture:
2 inputs → 4 hidden (ReLU) → 1 output (Sigmoid).
Predictions are random noise for now — training comes next.

## Step 7 — Loss function
Implemented MSE (Mean Squared Error) loss and its derivative in a static utility class.
Loss measures how wrong the network is — 0.0 is perfect, ~0.25 is random guessing.
The derivative tells backprop which direction to nudge the output to reduce error.
Untrained XOR network scores ~0.25 loss. Goal after training: below 0.01.

## Step 8 — Backpropagation and training (debugging)
Hit two bugs: sigmoid_derivative was receiving raw z instead of activated output,
and ReLU caused dying neurons on the small XOR dataset.
Fixed by passing sigmoid(z) into the derivative, and switching hidden layer to sigmoid.
Final result: loss 0.27 → 0.004 over 5000 epochs. All four XOR predictions correct.
ReLU remains implemented for larger problems where it outperforms sigmoid.

## Step 9 — Weight serialization
Built a binary serializer that saves and loads all network weights and biases
to a .bin file using raw byte I/O with reinterpret_cast.
A fresh untrained network loaded with saved weights predicts identically
to the original trained network — lossless round trip confirmed.
Added .gitignore to exclude generated binaries from version control.

## Step 10 — Project complete
Polished README with full architecture documentation and build instructions.
Published to LinkedIn. Project represents a complete ML engine built from
scratch: matrix math → activations → layers → training → serialization.
Total: 10 steps, zero external dependencies, one working neural network.