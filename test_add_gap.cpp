#include <iostream>
#include "cnn_layers.hpp"
#include <vector>

using namespace std;

int main(){
    Tensor3D test_imga ={
        {
            {1.0f, 2.0f},
            {3.0f, 4.0f}
        }
    };
    Tensor3D test_imgb ={
        {
            {10.0f, 20.0f},
            {30.0f, 40.0f}
        }
    };
    Tensor3D add_output;
    Tensor1D gap_output;
    Add adder;
    GlobalAvgPool gap;

    adder.forward(test_imga, test_imgb, add_output);

    gap.forward(test_imga, gap_output);

    //in ra kqua gap va add
    for(int h=0; h<2; h++){
        for(int w=0; w<2; w++){
            cout << add_output[0][h][w] << "\t";
        }
    }
    cout << endl;
    for(int i=0; i<gap_output.size(); i++){
        cout << gap_output[i] << "\t";
    }
}
