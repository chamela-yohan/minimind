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