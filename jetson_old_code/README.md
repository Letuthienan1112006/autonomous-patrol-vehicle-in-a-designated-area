# Jetson old code (archive)

Các file code cũ chỉ có trên Jetson Nano của dự án AGV, không nằm trên nhánh `main`: bản sao lưu trước mỗi lần sửa
(`*_before_*.py`, `*.bak*`), các script thử nghiệm (`test*.py`, `camera.py`, `daydu.py`, `car_test`, `AGV/`) và
các bản trùng tên nhưng khác nội dung (`variants/`).

Đã loại bỏ cố ý: thông tin bí mật (`secrets_local.py`), log, cơ sở dữ liệu, ảnh, bộ dữ liệu huấn luyện và trọng số mô hình.
Dữ liệu nặng được lưu riêng trên ổ SSD của Jetson, không đưa lên GitHub.
