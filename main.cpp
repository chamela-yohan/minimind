#include <iostream>
#include "src/math/matrix.h"

using namespace std;

int main(){
    cout << "MiniMind - Neural Network Engine" << endl;
    
    // Create a 3x3 matrix filled with zeros
    Matrix m(3, 3);
    m.print("Empty 3x3");

    // Set some values
    m.set(0,0,1.0);
    m.set(1,1,5.0);
    m.set(2,2,9.0);
    m.print("\nDiagonal set");

    // Read a value back
    cout<<"\nValue at (1, 1): " << m.get(1,1) << "\n";

    // Bound checking
    try{
        m.get(100, 100);
    }catch(const exception& e){
        std::cout << "Caught expected error: " << e.what() << "\n";
    }

    return 0;
}