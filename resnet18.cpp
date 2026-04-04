#include "resnet18.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>

//1. Hàm khởi tạo toàn bộ mạng dư
ResNet18::ResNet18(int num_classes)
    //Khởi tạo mạng đầu vào: 3 kênh vào, 64 kênh ra, kernel 3, stride 1, padding 1
    : conv1(3, 64, 3, 1, 1),
      bn1(64),

      // Layer 1: Giữ nguyên 64 kênh, stride = 1
      layer1_block1(64, 64, 1),
      layer1_block2(64, 64, 1),

      // Layer 2: Tăng lên 128 kênh. Block đầu stride 2 để giảm kích thước ảnh
      layer2_block1(64, 128, 2),
      layer2_block2(128, 128, 1),

      // Layer 3: Tăng lên 256 kênh. Block đầu stride 2 để giảm kích thước ảnh
      layer3_block1(128, 256, 2),
      layer3_block2(256, 256, 1),

      // Layer 4: Tăng lên 512 kênh. Block đầu stride 2 để giảm kích thước ảnh
      layer4_block1(256, 512, 2),
      layer4_block2(512, 512, 1),

      //Nhận 512 đặc trưng và xuất ra số lượng nhãn
      fc(512, num_classes)
{

}

void ResNet18::forward(const Tensor3D& input, Tensor1D& output) const {
    //Mảng trung gian
    Tensor3D x1, x2;

    //Cụm đầu vào
    conv1.forward(input, x1);
    bn1.forward(x1, x2);
    relu.forward(x2);

    //Layer1
    layer1_block1.forward(x2, x1);
    layer1_block2.forward(x1, x2);

    //Layer2
    layer2_block1.forward(x2, x1);
    layer2_block2.forward(x1, x2);

    //Layer3
    layer3_block1.forward(x2, x1);
    layer3_block2.forward(x1, x2);

    //Layer4
    layer4_block1.forward(x2, x1);
    layer4_block2.forward(x1, x2);

    Tensor1D o_features;
    avgpool.forward(x2, o_features);
    fc.forward(o_features, output);
}

int ResNet18::predict(const Tensor3D& input) const{
    Tensor1D score;

    //Đẩy ảnh vào mạng rồi lấy bảng điểm
    forward(input, score);

    //Tìm index điểm cao nhất
    int best_class = 0;
    float max_score = score[0];

    for(int i = 0; i < score.size(); ++i) {
        if (score[i] > max_score) {
            max_score = score[i];
            best_class = i;
        }
    }
    return best_class;
}

//2. Các hàm phụ trợ (dùng để đọc dữ liệu từ file vào mảng)
// Hàm hút trọng số cho Conv2D
void load_conv(Conv2D& conv, std::ifstream& file) {
    // Conv2D trong cấu trúc này không dùng bias, nên ta chỉ hút weight
    for (int out_c = 0; out_c < conv.out_channels; ++out_c) {
        for (int in_c = 0; in_c < conv.in_channels; ++in_c) {
            for (int h = 0; h < conv.kernel_size; ++h) {
                // Đọc 1 dòng (row) của kernel vào RAM
                file.read(reinterpret_cast<char*>(conv.weight[out_c][in_c][h].data()),
                          conv.kernel_size * sizeof(float));
            }
        }
    }
}
//Hút xong vào conv.weight

// Hàm hút trọng số cho BatchNorm
void load_bn(BatchNorm& bn, std::ifstream& file) {
    // Thứ tự lưu của PyTorch: weight(gamma) -> bias(beta) -> running_mean -> running_var
    file.read(reinterpret_cast<char*>(bn.weight.data()),       bn.num_features * sizeof(float));
    file.read(reinterpret_cast<char*>(bn.bias.data()),         bn.num_features * sizeof(float));
    file.read(reinterpret_cast<char*>(bn.running_mean.data()), bn.num_features * sizeof(float));
    file.read(reinterpret_cast<char*>(bn.running_var.data()),  bn.num_features * sizeof(float));
}

// Hàm hút trọng số cho Khối ResidualBlock
void load_block(ResidualBlock& block, std::ifstream& file) {
    // Hút nhánh chính
    load_conv(block.conv1, file);
    load_bn(block.bn1, file);
    load_conv(block.conv2, file);
    load_bn(block.bn2, file);

    // Hút nhánh phụ (nếu có bật)
    if (block.use_shortcut) {
        load_conv(block.shortcut_conv, file);
        load_bn(block.shortcut_bn, file);
    }
}

// 3. Hàm nạp trọng số vào bộ nhớ RAM
bool ResNet18::load_weight(const std::string& filepath) {
    // 1. Mở file nhị phân
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "LỖI: Không tìm thấy file trọng số " << filepath << "!\n";
        return false;
    }

    std::cout << "Đang truyền tải trọng số vào RAM ...\n";

    // 2. Hút Cụm đầu vào
    load_conv(conv1, file);
    load_bn(bn1, file);

    // 3. Hút Cụm thân (4 Tầng x 2 Block)
    load_block(layer1_block1, file);
    load_block(layer1_block2, file);
    
    load_block(layer2_block1, file);
    load_block(layer2_block2, file);
    
    load_block(layer3_block1, file);
    load_block(layer3_block2, file);
    
    load_block(layer4_block1, file);
    load_block(layer4_block2, file);

    // 4. Hút Cụm chốt hạ (Linear/FC)
    for (int o = 0; o < fc.out_features; ++o) {
        file.read(reinterpret_cast<char*>(fc.weight[o].data()), fc.in_features * sizeof(float));
    }
    file.read(reinterpret_cast<char*>(fc.biases.data()), fc.out_features * sizeof(float));

    // 5. Kiểm tra an toàn xem có hút thiếu hay không
    if (file.fail()) {
        std::cerr << "CẢNH BÁO: Quá trình đọc file bị gián đoạn giữa chừng!\n";
        file.close();
        return false;
    }

    file.close();
    std::cout << "HOÀN TẤT: Não bộ ResNet-18 đã sẵn sàng trực chiến!\n";
    return true;
}