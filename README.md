# 🫀 Heart Failure Project

Dự án được xây dựng nhằm thiết lập một hệ thống khép kín từ khâu tiền xử lý dữ liệu lâm sàng đến xây dựng mô hình toán học dự đoán nguy cơ suy tim dựa trên 918 mẫu. Hệ thống được cấu trúc module hóa chuyên nghiệp qua 3 cấu phần cốt lõi: xử lý mảng và thống kê mô tả nền tảng bằng NumPy, hiện thực hóa các giải thuật đại số và xác suất cho AI bằng cách kiểm tra hạng ma trận, phân tích giá trị suy biến SVD, tính toán xác suất Bayes và tự triển khai giải thuật tối ưu lặp Gradient Descent thủ công, cuối cùng là ứng dụng Pandas để làm sạch dị biệt dữ liệu, bóc tách mối quan hệ y khoa và áp dụng mã hóa One-Hot Encoding chuyên sâu. Toàn bộ pipeline không chỉ giải quyết bài toán phân lớp một cách tường minh, chặt chẽ về mặt toán học mà còn tự động xuất ra hệ thống biểu đồ trực quan.
---

# 📁 Cấu trúc thư mục - Directory Structure


```text
heart-failure-project/
├── data/
│   ├── raw/                  # Chứa file dữ liệu gốc ban đầu (heart.csv)
│   └── processed/            # Tập dữ liệu sạch sau khi xử lý và mã hóa
├── notebooks/
│   └── exploratory_analysis.ipynb  # Notebook chạy thực nghiệm trực quan
├── src/                      # Mã nguồn phân rã theo module chức năng
│   ├── data_processing.py    # Xử lý mảng NumPy & Thống kê mô tả (Module 1)
│   ├── feature_engineering.py # Kỹ nghệ đặc trưng dữ liệu & Làm sạch (Module 3)
│   ├── modelling.py          # Triển khai giải thuật Toán & Gradient Descent (Module 2)
│   └── utils.py              # Các hàm phụ trợ hệ thống
├── outputs/                  # Kết quả tự động sinh ra từ hệ thống
│   ├── figures/              # Biểu đồ phân phối, Boxplot, Đường hội tụ Loss
│   ├── tables/               # Các bảng thống kê Pivot Table dạng .csv
├── report/
├── requirements.txt          # Danh sách thư viện phụ thuộc của hệ thống
└── README.md                 # Hướng dẫn tổng quan dự án (File này)
```
---
# ⚙️ Cài đặt - Setup
Dự án sử dụng Anaconda để thiết lập và cô lập môi trường ảo, tránh xung đột thư viện và sử dụng PyCharm làm môi trường phát triển (IDE).
## Tạo và kích hoạt môi trường bằng Anaconda
```bash
# Tạo môi trường ảo mới tên là 'heart_env' sử dụng Python 3.10.20
conda create --name heart_env python=3.10.20 -y

# Kích hoạt môi trường vừa tạo
conda activate heart_env

# Cài đặt các gói thư viện bắt buộc từ requirements.txt
pip install -r requirements.txt
```

## Cấu hình Interpreter trên PyCharm

Để PyCharm nhận diện đúng môi trường Conda vừa tạo, bạn thực hiện cấu hình theo các bước sau:

1. Mở thư mục dự án bằng PyCharm.

2. Đi đến đường dẫn: File -> Settings (hoặc Preferences trên macOS) -> Project: heart-failure-project -> Python Interpreter.

3. Kích vào biểu tượng bánh răng ⚙️ hoặc chọn Add Interpreter... -> Add Local Interpreter...

4. Chọn mục Conda Environment ở thanh menu bên trái.

5. Chọn mục Use existing environment và tìm đến môi trường heart_env trong danh sách xổ xuống.

6. Nhấn OK và Apply để hoàn tất cấu hình.

## Hướng dẫn khởi chạy dự án

1. Trong giao diện PyCharm, mở file notebooks/exploratory_analysis.ipynb.

2. Đảm bảo góc phải màn hình Notebook đã chọn đúng Interpreter là heart_env.

3. Nhấn nút Run All (hoặc tổ hợp phím Shift + Enter trên từng ô code) để thực thi toàn bộ pipeline dữ liệu.

4. Hệ thống sẽ tự động gọi mã nguồn từ thư mục src/ để thực hiện tính toán thống kê, vẽ biểu đồ phân phối và tối ưu hóa toán học.
