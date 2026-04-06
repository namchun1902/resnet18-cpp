/**
 * File cifar10_io.cpp
 * @brief Triển khai hàm đọc và xử lý tập dữ liệu CIFAR-10
 */

#include "cifar10_io.hpp"
#include <fstream>
#include <iostream>

std::vector<Image> read_cifar10(const std::string& filepath) {
    std::vector<Image> dataset;

    //Mở file chế độ đọc nhị phân
    std::ifstream file (filepath, std::ios::binary);
    if(!file.is_open()) {
        std::cerr << "Lỗi: Không thể mở file " << filepath << std::endl;
        return dataset;   //Nếu không mở được thì trả về mảng rỗng
    }

    //Khai báo hằng số theo định dạng CIFAR-10
    constexpr int num_images = 10000;
    constexpr int image_size = 3072;

    //Ép cấp phát sẵn bộ nhớ 10.000 ảnh
    dataset.reserve(num_images);

    //Bắt đầu vòng lặp lấy từng ảnh ra từ file
    for (int i = 0; i < num_images; i++){
        unsigned char label;

        //Vừa đọc vừa kiểm tra xem file có thiếu ảnh không
        if(!file.read(reinterpret_cast<char*>(&label), 1)) {
            std::cerr << "File lỗi hoặc kết thúc sớm ở ảnh " << i << std::endl;
            break;
        }

        Image img;
        img.label = static_cast<int>(label);
        img.data.assign(3, Tensor2D(32, Tensor1D(32, 0.0f)));
    
        //Đọc và đổ dữ liệu ảnh vào buffer
        unsigned char buffer[image_size];
        file.read(reinterpret_cast<char*>(buffer), image_size);

        //Đổ và chuẩn hoá dữ liệu ảnh vào img.data
        int index = 0;
        for (int c = 0; c < 3; c++){
            for (int h = 0; h < 32; h++){
                for (int w = 0; w < 32; w++){
                    img.data[c][h][w] = static_cast<float>(buffer[index]) / 255.0f;
                    index++;
                }
            }
        }

    //Đẩy ảnh đã xử lý xong vào cuối dataset
    dataset.push_back(std::move(img));
    }

    //Đóng file
    file.close();
    std::cout << "Đã tải hết ảnh vào bộ nhớ RAM " << std::endl;

    return dataset;
}