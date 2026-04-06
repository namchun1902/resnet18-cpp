#include <iostream>
#include "cnn_layers.hpp"
#include <vector>  

using namespace std;

int main(){
    cout << "Kiem tra lop BatchNorm\n";

    //1. Anh 2 kenh mau kich thuoc 2x2
    Tensor3D test_image = {
        {
            {10.0f, 10.0f},
            {10.0f, 10.0f}
        },
        {
            {-5.0f, 0.0f},
            {5.0f, 10.0f}
        }
    };
    //kenh 0
    BatchNorm test_bn(2);
    test_bn.running_mean[0]= 5.0f;
    test_bn.running_var[0]= 4.0f;  //std=2
    test_bn.weight[0]= 2.0f;
    test_bn.bias[0]= 1.0f;

    //kenh 1
    test_bn.running_mean[1]= 0.0f;
    test_bn.running_var[1]= 1.0f;  //std=1
    test_bn.weight[1]= 1.0f;
    test_bn.bias[1]= -2.0f;

    Tensor3D output;
    test_bn.forward(test_image, output);

    //In ket qua, tat ca diem anh ky vong la 6.00
    for(int h=0; h<2; h++){
        for(int w=0; w<2; w++){
            cout << output[0][h][w] << "\t";
        }
        cout << endl << endl;
    }

    //In ra ket qua, ky vong -7.00 -2.00 3.00 8.00
    for(int h=0; h<2; h++){
        for(int w=0; w<2; w++){
            cout << output[1][h][w] << "\t";
        }
        cout << endl;
    }
}
