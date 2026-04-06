//Test Conv2D
#include <iostream>
#include "cnn_layers.hpp"
#include <vector>

using namespace std;

int main(){
    cout<< "Kiem tra lop Conv2D\n";

    Tensor3D test_image = {
        {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
        }
    };

    Conv2D test_conv(1, 1, 2, 1, 0); // in=out=stride=1, kernel 2x2, padding=0
    //  Kernel: [ [1,  0], 
    //            [0, -1] ]
    test_conv.weight[0][0][0][0] = 1.0f;
    test_conv.weight[0][0][1][0] = 0.0f;
    test_conv.weight[0][0][0][1] = 0.0f;
    test_conv.weight[0][0][1][1] = -1.0f;

    Tensor3D output;
    test_conv.forward(test_image, output);
    cout << "ma tran ket qua" << endl;

    for( int h = 0; h < output[0].size(); h++){
        for( int w = 0; w < output[0][h].size(); w++){
            cout << output[0][h][w] << "\t";
        }
        cout << endl;
    }
    return 0;
}
