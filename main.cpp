#include <iostream>
#include "src/math/matrix.h"

using namespace std;

int main()
{
    cout << "MiniMind - Neural Network Engine" << endl;

    // Add
    Matrix a(2, 2);
    a.set(0, 0, 1);
    a.set(0, 1, 2);
    a.set(1, 0, 3);
    a.set(1, 1, 4);

    Matrix b(2, 2);
    b.set(0, 0, 10);
    b.set(0, 1, 20);
    b.set(1, 0, 30);
    b.set(1, 1, 40);

    a.add(b).print("Add: a + b");

    // Scale
    a.scale(3.0).print("\nScale: a * 3");

    // Transpose
    Matrix c(2, 3);
    c.set(0, 0, 1);
    c.set(0, 1, 2);
    c.set(0, 2, 3);
    c.set(1, 0, 4);
    c.set(1, 1, 5);
    c.set(1, 2, 6);
    c.print("\nOriginal c (2x3)");
    c.transpose().print("Transposed c (3x2)");

    // Multiply
    Matrix A(2, 3);
    A.set(0, 0, 1);
    A.set(0, 1, 2);
    A.set(0, 2, 3);
    A.set(1, 0, 4);
    A.set(1, 1, 5);
    A.set(1, 2, 6);

    Matrix B(3, 2);
    B.set(0, 0, 7);
    B.set(0, 1, 8);
    B.set(1, 0, 9);
    B.set(1, 1, 10);
    B.set(2, 0, 11);
    B.set(2, 1, 12);

    A.multiply(B).print("\nMultiply: A x B (expect 58,64 / 139,154)");

    // Dimension mismatch guard
    try
    {
        a.multiply(b.transpose().transpose().transpose()); // force a bad multiply
        a.add(c);
    }
    catch (const std::exception &e)
    {
        std::cout << "\nCaught expected error: " << e.what() << "\n";
    }
}