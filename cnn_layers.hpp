/**
 * @file cnn_layers.hpp
 * @brief Định nghĩa các cấu trúc dữ liệu và các lớp xử lý cốt lõi cho mạng Resnet-18
 */

#pragma once
#include <vector>

using Tensor1D = std::vector<float>;
using Tensor2D = std::vector<Tensor1D>;
using Tensor3D = std::vector<Tensor2D>;
using Tensor4D = std::vector<Tensor3D>;

/**
 * @struct Conv2D
 * @brief Lớp tích chập 2 chiều.
 * @details Thực hiện phép nhân chập để trích xuất đặc trưng. 
 */
struct Conv2D{
    int in_channels;  //Số kênh đầu vào, số kênh Tensor đầu vào = số kernel 1 filter
    int out_channels; //Số kênh đầu ra = số filter
    int kernel_size;  //Kích thước bộ lọc
    int stride;       //Bước trượt
    int padding;      //Padding

    //Learnable Parameters
    //Kích thước weight: [out_channels][in_channels][kernel_size][kernel_size]
    Tensor4D weight;

    //Kích thước biases: [out_channels]
    Tensor1D biases;

    //Hàm khởi tạo (Constructor)
    Conv2D(int in_c, int out_c, int k, int s, int p);

    /**
     * @brief Lan truyền tiến (Forward pass)
     * @param input Tensor3D đầu vào
     * @param output Tensor3D đầu ra (truyền tham chiếu để ghi đè kết quả)
     */
    void forward(const Tensor3D& input, Tensor3D& output) const;
};


/**
 * @struct BatchNorm
 * @brief Lớp chuẩn hoá batch
 * @details Chuẩn hoá dữ liệu trong mỗi kênh về phân phối chuẩn
 * sau đó dùng tham số học được (gamma, beta) để co giãn và dịch chuyển lại dữ liệu
 */
struct BatchNorm {
    //Số kênh đầu vào( = số kênh đầu ra)
    int num_features;

    //Learnable Parameters
    //Kích thước: [num_features]
    Tensor1D weight; //Hệ số scale (gamma)
    Tensor1D bias;   //Hệ số dịch chuyển (beta)

    //Các thống kê từ training
    Tensor1D running_mean; //Trung bình độ sáng từng kênh
    Tensor1D running_var;  //Phương sai từng kênh
    const float eps = 1e-5f;

    //Hàm khởi tạo (Constructor)
    BatchNorm(int num_features);

    /**
     * @brief Lan truyền tiến (Forward pass)
     * @param input Tensor3D đầu vào [C][H][W]
     * @param output Tensor3D đầu ra (truyền tham chiếu để ghi đè kết quả)
     */
    void forward(const Tensor3D& input, Tensor3D& output) const;
};


/**
 * @struct ReLU
 * @brief Hàm kích hoạt phi tuyến 
 * @details Cắt bỏ giá trị âm
 */
struct ReLU {
    /**
     * @brief Lan truyền tiến (Forward pass)
     * @param input Tensor3D đầu vào (Truyền tham chiếu không có const để ghi đè dữ liệu)
     */
    void forward(Tensor3D& tensor) const;
};

/**
 * @struct Add
 * @brief Lớp tính tổng 2 Tensor
 * @details Tính tổng nhánh chính và nhánh phụ
 */
struct Add {
    /**
     * @brief Lan truyền tiến (Forward pass)
     * @param input input_a Tensor nhánh chính
     * @param input input_b Tensor nhánh phụ
     * @param output Tensor kết quả phép cộng
     */
    void forward(const Tensor3D& input_a, const Tensor3D& input_b, Tensor3D& output) const;
};



/**
 * @struct Residual Block
 * @brief Khối residual cơ bản của mạng Resnet-18
 * @details Bao gồm nhánh chính và nhánh tắt để thực hiện skip connection
 */
struct ResidualBlock {
    //Các lớp thành phần của nhánh chính: 2 conv, 2 bn, 0 relu vì stateless
    Conv2D conv1; //Lớp tích chập đầu
    BatchNorm bn1;//Lớp chuẩn hoá batch đầu tiên
    ReLU relu1; //ReLU giữa nhánh chính
    Conv2D conv2; //Lớp tích chập thứ hai
    BatchNorm bn2;//Lớp chuẩn hoá batch thứ hai

    //Các biến và lớp thành phần của nhánh tắt
    bool use_shortcut;
    Conv2D shortcut_conv;
    BatchNorm shortcut_bn;

    //Trạm xử lý cuối
    Add adder;
    ReLU relu2; //ReLU chốt sau khi cộng

    //Hàm khởi tạo (Constructor)
    ResidualBlock(int in_c, int out_c, int stride = 1);
    //Hàm lan truyền tiến (Forward pass)
    // @param input Tensor3D đầu vào
    void forward(const Tensor3D& input, Tensor3D& output) const;
};


/**
 * @struct GlobalAvgPool
 * @brief Lớp pooling trung bình toàn cục
 * @details Ép 3D thành 1D
 * Mỗi kênh tính trung bình toàn bộ điểm ảnh trong kênh
 */
struct GlobalAvgPool {
    /**
     * @brief Lan truyền tiến (Forward pass)
     * @param input Tensor3D đầu vào (ví dụ [512][4][4])
     * @param output Tensor1D đầu ra (ví dụ [512])
     */
    void forward(const Tensor3D& input, Tensor1D& output) const;
};


/**
 * @struct Linear (Full Connected)
 * @brief  Lớp kết nối đầy đủ
 * @details Đóng vai trò phân loại ở cuối mạng
 * Kết nối mọi đặc trưng của ảnh vào một vector 1D
 */
struct Linear {
    int in_features; //Số đặc trưng đầu vào(Vd: 512)
    int out_features; //Số đặc trưng đầu ra(Vd: 10)

    //Learnable Parameters
    //Kích thước weight: [out_features][in_features]
    Tensor2D weight;

    //Kích thước biases: [out_features]
    Tensor1D biases;

    //Hàm khởi tạo (Constructor)
    Linear(int in, int out);

    /**
     * @brief Lan truyền tiến (Forward pass)
     * @param input Tensor1D đầu vào (ví dụ: [512])
     * @param output Tensor1D đầu ra (ví dụ: [10])
     */
    void forward(const Tensor1D& input, Tensor1D& output) const;
};