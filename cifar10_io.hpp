/**
   *File: cifar10_io.hpp
   *Chức năng: Khai báo cấu trúc dữ liệu và các hàm
   *phục vụ việc trích xuất tập dữ liệu CIFAR-10 từ định dạng nhị phân
*/

#pragma once

#include <vector>
#include <string>
#include "cnn_layers.hpp"

/**
 * @struct Image 
 * @brief Cấu trúc lưu trữ thông tin của một bức ảnh trong tập CIFAR-10.
 *  Mỗi bức ảnh trong CIFAR-10 có kích thước 32x32x3 (3 kênh màu RGB).
 */
struct Image {
    // Nhãn thực tế của bức ảnh (giá trị từ 0 đến 9 ứng với 10 lớp vật thể)
    int label;

    // Khối dữ liệu 3D chứa điểm ảnh: [3 kênh][32 chiều cao][32 chiều rộng]
    // Đã được chuẩn hoá về dải [0.0, 1.0]
    Tensor3D data;
};

/**
 * @brief Đọc dữ liệu từ file binary của tập CIFAR-10.
 * Hàm thực hiện đọc tuần tự cấu trúc 3073 byte của mỗi bức ảnh
 * (1 byte là label, 3072 bytes là dữ liệu ảnh RGB)
 * @param filepath Đường dẫn tới file nhị phân.
 * @return std::vector<Image> Mảng vector chứa toàn bộ các bức ảnh đã được bóc tách
 */
std::vector<Image> read_cifar10(const std::string& filepath);