/**
 * @file main.cpp
 * @brief Chương trình chính (Main) - Kết nối module I/O và mạng ResNet-18 để chạy suy luận (Inference)
 */

#include <iostream>
#include <vector>
#include <string>
#include "cifar10_io.hpp"
#include "resnet18.hpp"

// Hỗ trợ hiển thị tiếng Việt trên Windows Console
#ifdef _WIN32
#include <windows.h>
#endif

int main() {
    // 1. Cấu hình môi trường
    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    #endif

    std::cout << "========================================================\n";
    std::cout << "      CHƯƠNG TRÌNH SUY LUẬN RESNET-18 (C++ THUẦN)\n";
    std::cout << "========================================================\n\n";

    // 2. Khởi tạo kiến trúc mạng
    std::cout << "[INFO] Đang khởi tạo kiến trúc mạng ResNet-18...\n";
    ResNet18 model(10); // Khởi tạo mô hình với 10 lớp phân loại (classes)

    // 3. Nạp trọng số mô hình (Load Weights)
    std::string weight_path = "D:/resnet18/resnet18_weights.bin"; 
    
    if (!model.load_weight(weight_path)) {
        std::cerr << "[ERROR] Khởi tạo thất bại do không nạp được tệp trọng số!\n";
        return -1; // Kết thúc chương trình với mã lỗi
    }

    // 4. Tải tập dữ liệu kiểm thử (Load Test Dataset)
    std::string data_path = "D:/resnet18/cifar-10-batches-bin/test_batch.bin";
    std::vector<Image> dataset = read_cifar10(data_path);
    
    if (dataset.empty()) {
        std::cerr << "[ERROR] Không thể nạp tập dữ liệu ảnh!\n";
        return -1;
    }

    // Từ điển nhãn của CIFAR-10
    const std::string class_names[] = {
        "Máy bay (Airplane)", "Ô tô (Automobile)", "Chim (Bird)", "Mèo (Cat)",
        "Hươu (Deer)", "Chó (Dog)", "Ếch (Frog)", "Ngựa (Horse)",
        "Tàu thủy (Ship)", "Xe tải (Truck)"
    };

    // 5. Chạy suy luận (Inference) trên ảnh thực tế
    std::cout << "\n========================================================\n";
    std::cout << "                BẮT ĐẦU CHẠY SUY LUẬN                   \n";
    std::cout << "========================================================\n";

    // Chọn ảnh ở vị trí số 0 trong tập test
    int test_idx = 0; 
    Image& test_img = dataset[test_idx];

    std::cout << "[ACTION] Đang thực hiện lan truyền tiến (Forward Pass) cho ảnh số " << test_idx << "...\n";
    
    // Đưa ảnh qua mạng để lấy dự đoán
    int predicted_class = model.predict(test_img.data);
    int true_class = test_img.label;

    // In kết quả đối chiếu
    std::cout << "\n---------------- KẾT QUẢ ----------------\n";
    std::cout << "  -> Nhãn thực tế (Ground Truth) : " << class_names[true_class] << "\n";
    std::cout << "  -> Mô hình dự đoán (Prediction): " << class_names[predicted_class] << "\n";
    std::cout << "-----------------------------------------\n";

    if (predicted_class == true_class) {
        std::cout << "  => [KẾT QUẢ] Nhận diện CHÍNH XÁC!\n";
    } else {
        std::cout << "  => [KẾT QUẢ] Nhận diện SAI.\n";
    }

    std::cout << "\n[DONE] Quá trình thực thi hoàn tất.\n";
    return 0;
}