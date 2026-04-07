/**
 * @file resnet18.hpp
 * @brief Top module của ResNet18
 * @details ảnh [3x32x32] -> cụm 1 -> 4 tầng ResBlock -> GAP -> Linear 10 nhãn
 */

 #pragma once
 #include "cnn_layers.hpp"
 #include <string>

struct ResNet18 {
    //1. Cụm đầu vào
    // Chuyển ảnh 3 kênh màu thành 64 kênh đặc trưng
    Conv2D conv1;
    BatchNorm bn1;
    ReLU relu;
    MaxPool2D maxpool;

    //2. Cụm chính
    // Bao gồm 4 tầng, mỗi tầng 2 resblock

    //Layer 1: tăng lên 64 kênh, giữ nguyên độ phân giải 32x32
    ResidualBlock layer1_block1;
    ResidualBlock layer1_block2;
    
    //Layer 2: tăng lên 128 kênh, ép độ phân giải 16x16
    ResidualBlock layer2_block1;
    ResidualBlock layer2_block2;

    // Layer 3: tăng lên 256 kênh, ép độ phân giải 8x8
    ResidualBlock layer3_block1;
    ResidualBlock layer3_block2;

    // Layer 4: tăng lên 512 kênh, ép độ phân giải 4x4
    ResidualBlock layer4_block1;
    ResidualBlock layer4_block2;

    //3. Cụm phân loại
    // Ép thành mảng 1D và chấm điểm
    GlobalAvgPool avgpool;
    Linear fc;

    //4. Các hàm giao tiếp

    //Hàm khởi tạo toàn bộ mạng dư
    ResNet18(int num_classes = 10);

    //Hàm đọc dữ liệu trọng số file bin
    bool load_weight(const std::string& filepath);

    //Hàm lan truyền tiến 1 bức ảnh
    void forward(const Tensor3D& input, Tensor1D& output) const;

    //Hàm trả về ID nhãn có điểm số cao nhất
    int predict(const Tensor3D& input) const;
};