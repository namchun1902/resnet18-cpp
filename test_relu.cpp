#include <iostream>
#include "cnn_layers.hpp"
using namespace std;

int main(){
    Tensor3D test_image = {
        {
            {-5.5f,  0.0f,  3.2f},
            { 1.1f, -2.0f, -0.1f},
            {10.0f, -9.9f,  0.5f}
        }
    };
    ReLU test_relu;
    test_relu.forward(test_image);
    //In ket qua, tat ca diem anh ky vong la 0.00
    for(int h=0; h<3; h++){
        for(int w=0; w<3; w++){
            cout << test_image[0][h][w] << "\t";
        }
        cout << endl;
    }
    return 0;
}