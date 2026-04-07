#include <iostream>
#include <vector>
#include <string>
#include "cifar10_io.hpp"
#include "resnet18.hpp"
#include <windows.h>

using namespace std;

int main(){
    //Cấu hình môi trường
    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    #endif


    ResNet18 model(10);
    string weight_path = "D:/resnet18/resnet18_weights.bin";
    model.load_weight(weight_path);

    string data_path = "D:/resnet18/cifar-10-batches-bin/test_batch.bin";
    vector<Image> dataset = read_cifar10(data_path);

    int test_max = 10;
    int correct_count = 0;

    for(int i=0; i<test_max; i++){
        int predicted_class = model.predict(dataset[i].data);
        if(predicted_class == dataset[i].label) correct_count++;
    }
    cout << "KET QUA CUOI CUNG: " << correct_count << " / " << test_max << endl;

    return 0;
}
