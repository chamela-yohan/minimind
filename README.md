# MiniMind

A neural network engine built from scratch in C++ — no ML libraries, 
no PyTorch, no shortcuts. Every matrix operation, activation function, 
backpropagation step, and weight update is hand-written.

---

## Why this exists

Everyone uses neural networks. Almost nobody understands what's inside them.
This project is about understanding from the ground up — in C++, the same 
language PyTorch itself is written in.

---

## What it does

- Trains a 2-layer neural network to solve the XOR problem
- Implements forward pass, backpropagation, and gradient descent from scratch
- Saves and loads trained weights to a binary `.bin` file
- Live ASCII loss curve in the terminal during training

**Result:** Loss drops from ~0.27 (random guessing) to ~0.004 (96%+ accuracy) over 5000 epochs.

```
Epoch     0 | Loss: 0.271117 [##########                              ]
Epoch  2500 | Loss: 0.120651 [####                                    ]
Epoch  5000 | Loss: 0.004458 [                                        ]

Final predictions:
  [0,0] → 0.054  (expected 0) ✓
  [0,1] → 0.925  (expected 1) ✓
  [1,0] → 0.943  (expected 1) ✓
  [1,1] → 0.077  (expected 0) ✓
```

---

## Architecture

```
Input (2)
   ↓
Dense Layer — 2→4 neurons, Sigmoid activation
   ↓
Dense Layer — 4→1 neurons, Sigmoid activation
   ↓
Prediction (single value 0–1)
```

Built in layers — each one independent and testable:

```
src/
├── math/
│   ├── matrix.h        — Matrix class: storage, operations, dot product
│   └── activations.h   — Sigmoid, ReLU and their derivatives
├── network/
│   ├── layer.h         — DenseLayer: forward + backward pass
│   └── network.h       — Network: chains layers together
├── training/
│   ├── loss.h          — MSE loss and derivative
│   └── trainer.h       — Training loop with live loss display
└── io/
    └── serializer.h    — Binary weight save/load
```

---

## Build and run

```bash
g++ main.cpp -o minimind
./minimind
```

No dependencies. Just a C++ compiler.

---

## Key concepts implemented

| Concept | Where |
|---|---|
| Matrix dot product | `src/math/matrix.h` |
| Xavier weight initialization | `src/network/layer.h` |
| Sigmoid / ReLU + derivatives | `src/math/activations.h` |
| Forward pass | `src/network/layer.h` |
| Backpropagation | `src/network/layer.h` — `backward()` |
| Gradient descent | `src/training/trainer.h` |
| Binary serialization | `src/io/serializer.h` |

---

## Dev log

See [DEVLOG.md](DEVLOG.md) for a step-by-step build journal.