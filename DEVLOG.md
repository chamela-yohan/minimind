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