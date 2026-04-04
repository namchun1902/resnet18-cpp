#include "cnn_layers.hpp"

#include <cmath>


//Định nghĩa cho hàm khởi tạo Conv2D của struct Conv2D
Conv2D::Conv2D(int in_c, int out_c, int k, int s, int p){
    in_channels = in_c;
    out_channels = out_c;
    kernel_size = k;
    stride = s;
    padding = p;


    //Cấp phát mảng weight 4D toàn số 0
    //Được nạp số thật vào sau
    weight.assign(out_channels, Tensor3D(in_channels, Tensor2D(kernel_size, Tensor1D(kernel_size, 0.0f))));

    //Cấp phát mảng biases toàn số 0
    biases.assign(out_channels, 0.0f);
}

//Hàm lan truyền tiến
void Conv2D::forward(const Tensor3D& input, Tensor3D& output) const {
    int h_in = input[0].size();
    int w_in = input[0][0].size();

    //Kích thước ảnh đầu ra
    int h_out = (h_in + 2*padding - kernel_size)/stride + 1;
    int w_out = (w_in + 2*padding - kernel_size)/stride + 1;

    //Cấp phát kích thước cho mảng output
    output.assign(out_channels, Tensor2D(h_out, Tensor1D(w_out, 0.0f)));

    //Quét trên output
    for(int out_c = 0; out_c < out_channels; out_c++){
        for(int h = 0; h < h_out; h++){
            for(int w = 0; w < w_out; w++){

                //Trượt filter trên ảnh đầu vào để tính pixel [h][w] của kênh out_c
                float sum = biases[out_c];

                //Cộng tất cả kernel lại
                for(int in_c = 0; in_c < in_channels; in_c++){
                    for(int kh = 0; kh < kernel_size; kh++){
                        for(int kw = 0; kw < kernel_size; kw++){

                            //Tính index pixel ảnh đầu vào ứng pixel [h][w] của đầu ra
                            int idx_h_in = h*stride + kh - padding;
                            int idx_w_in = w*stride + kw - padding;

                            //Kiểm tra index pixel có trong ảnh đầu vào hay không
                            if(idx_h_in >= 0 && idx_h_in < h_in && idx_w_in >= 0 && idx_w_in < w_in){
                                sum += input[in_c][idx_h_in][idx_w_in] * weight[out_c][in_c][kh][kw];
                            }
                        }
                    }
                }
                output[out_c][h][w] = sum; //Giá trị pixel [h][w] của kênh out_c
            }
        }
    }
}


//Định nghĩa cho hàm khởi tạo BatchNorm của struct BatchNorm

BatchNorm::BatchNorm(int num_features) : num_features(num_features) {
    weight.assign(num_features, 1.0f);      //Hệ số co giãn (gamma), mặc định 1.0 là không co giãn
    bias.assign(num_features, 0.0f);        //Hệ số dịch chuyển (beta), mặc định 0.0 là không dịch chuyển
    running_mean.assign(num_features, 0.0f);//Trung bình độ sáng từng kênh
    running_var.assign(num_features, 1.0f); //Phương sai (Mặc định 1 tránh chia 0)
}

//Hàm lan truyền tiến BN
//Thực hiện chuẩn hoá và điều chỉnh
void BatchNorm::forward(const Tensor3D& input, Tensor3D& output) const {
    int channels = input.size();
    int height = input[0].size();
    int width = input[0][0].size();

    //Cấp phát kích thước cho mảng output
    output.assign(channels, Tensor2D(height, Tensor1D(width, 0.0f)));

    for(int c = 0; c < channels ; c++){
        float mean = running_mean[c];
        float var = running_var[c];
        float gamma = weight[c];
        float beta = bias[c];

        for(int h = 0; h < height; h++){
            for(int w = 0; w < width; w++){
                float normalized = (input[c][h][w] - mean) / std::sqrt(var + eps);
                output[c][h][w] = gamma * normalized + beta;
            }
        }
    }
}

//Hàm lan truyền tiến ReLU
void ReLU::forward(Tensor3D& tensor) const {
    for(int c = 0; c < tensor.size(); c++){
        for(int h = 0; h < tensor[0].size(); h++){
            for(int w = 0; w < tensor[0][0].size(); w++){
                if(tensor[c][h][w] < 0.0f) tensor[c][h][w] = 0.0f;
            }
        }
    }
}

//Hàm lan truyền tiến Add
void Add::forward(const Tensor3D& input_a, const Tensor3D& input_b, Tensor3D& output) const {
    int channels = input_a.size();
    int height = input_a[0].size();
    int width = input_a[0][0].size();
    output.assign(channels, Tensor2D(height, Tensor1D(width, 0.0f)));
    for(int c = 0; c < channels; c++){
        for(int h = 0; h < height; h++){
            for(int w = 0; w < width; w++){
                output[c][h][w] = input_a[c][h][w] + input_b[c][h][w];
            }
        }
    }
}

//Hàm lan truyền tiến GAP
void GlobalAvgPool::forward(const Tensor3D& input, Tensor1D& output) const {
    int channels = input.size();
    int height = input[0].size();
    int width = input[0][0].size();
    output.assign(channels, 0.0f);
    for(int c = 0; c < channels; c++){
        float sum = 0.0f;
        for(int h = 0; h < height; h++){
            for(int w = 0; w < width; w++){
                sum += input[c][h][w];
            }
        }
        output[c] = sum / (height * width);
    }
}

//Hàm khởi tạo Linear
Linear::Linear(int in, int out){
    in_features = in;
    out_features = out;
    weight.assign(out_features, Tensor1D(in_features, 0.0f));
    biases.assign(out_features, 0.0f);
}

//Hàm lan truyền tiến Linear
void Linear::forward(const Tensor1D& input, Tensor1D& output) const{
    output.assign(out_features, 0.0f);
    for(int o = 0; o < out_features; o ++){
        float sum = biases[o];
        for(int i = 0; i < in_features; i ++){
            sum += input[i] * weight[o][i];
        }
        output[o] = sum;
    }
}

//Hàm khởi tạo ResBlock
ResidualBlock::ResidualBlock(int in_c, int out_c, int stride) 
    : conv1(in_c, out_c, 3, stride, 1),
      bn1(out_c),
      conv2(out_c, out_c, 3, 1, 1),
      bn2(out_c),
      use_shortcut((in_c != out_c) || (stride != 1)),
      shortcut_conv(in_c, out_c, 1, stride, 0),
      shortcut_bn(out_c)
{
    //K có gì
}

//Hàm lan truyền tiến ResBlock
void ResidualBlock::forward(const Tensor3D& input, Tensor3D& output) const {
    Tensor3D conv1_out, bn1_out, conv2_out, bn2_out, shortcut_out;
    conv1.forward(input, conv1_out);
    bn1.forward(conv1_out, bn1_out);
    relu1.forward(bn1_out);
    conv2.forward(bn1_out, conv2_out);
    bn2.forward(conv2_out, bn2_out);

    //Shortcut nếu có
    if(use_shortcut){
        Tensor3D temp_shortcut;
        shortcut_conv.forward(input, temp_shortcut);
        shortcut_bn.forward(temp_shortcut, shortcut_out);
    }
    else{
        shortcut_out = input;
    }
    adder.forward(bn2_out, shortcut_out, output);
    relu2.forward(output);
}
