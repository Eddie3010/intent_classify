# AI Sales Assistant - Intent Classification API

Dự án này chuyên phân loại ý định (intent) của khách hàng sử dụng mô hình Mạng nơ-ron nhân tạo (MLP) dựa trên PyTorch, và phục vụ trực tiếp qua Web API bằng FastAPI.

## Cấu trúc thư mục

- `ai_sales_intent_dataset_1000_samples.csv`: File dữ liệu thô (dataset) dùng để huấn luyện mô hình.
- `train.py`: Script tiền xử lý văn bản (TF-IDF, One-Hot Encoding) và huấn luyện mô hình phân loại intent dùng PyTorch.
- `app.py`: Source code của FastAPI server, sẽ load mô hình đã train để dự đoán ý định khách hàng từ HTTP Request.
- `requirements.txt`: Chứa danh sách các thư viện Python phụ thuộc.
- `model.pth`: File trọng số của mạng neural PyTorch (được sinh ra sau khi chạy train).
- `preprocessors.pkl`: File lưu trữ các bộ Vectorizers và Encoders như (TF-IDF, LabelEncoder) (sinh ra sau khi chạy train).
- Các file hình ảnh báo cáo (`loss_curve.png`, `confusion_matrix.png`, `classification_metrics.png`) sinh ra từ file `train.py`.

## Cài đặt mô trường

Yêu cầu: Máy tính cần có cài đặt sẵn Python (phiên bản 3.9 trở lên).

Khởi chạy terminal (Powershell hoặc Command Prompt) tại thư mục chứa source code và cài đặt các thư viện cần thiết bằng câu lệnh:

```bash
# Đối với Windows
py -m pip install -r requirements.txt
```

*(Ghi chú: Nếu hệ điều hành thuộc Linux/macOS, vui lòng đổi lệnh `py` thành `python` hoặc `python3`)*

## 1. Huấn luyện lại mô hình (Training)

Quá trình này cần thiết nếu bạn có sự thay đổi về file dữ liệu đầu vào.
Chạy file train bằng câu lệnh:

```bash
py train.py
```

Sau khi chạy xong, chương trình sẽ tự động trích xuất các thông số training (loss / epochs), thông số validation, đồng thời tạo ra và lưu 3 ảnh chụp chỉ số đánh giá đồ họa vào thư mục hiện tại. Các file cấu hình mô hình (`model.pth` và `preprocessors.pkl`) cũng được tạo lại để API Engine có thể gọi và sử dụng.

## 2. Server Dự Đoán bằng FastAPI (Serving API)

Để khởi động Server Backend FastAPI, bạn sử dụng lệnh:

```bash
py -m uvicorn app:app --reload
```

Server sẽ lắng nghe các request mặc định tại port `8000`.

### Tương tác với API
Truy cập UI tự động của hệ thống tại URL: **http://127.0.0.1:8000/docs** bằng trình duyệt của bạn (Swagger UI). Bạn có thể test trực tiếp các chức năng trên giao diện này.

### Endpoint: `POST /predict`

* Request Payload Mẫu:

  ```json
  {
    "previous_Intent": "product_inventory",
    "previous_bot_response": "Hiện tại tính năng còn mới",
    "current_query": "Chỉ chú shop gần đây với"
  }
  ```

* Response Mẫu:

  ```json
  {
    "current_intent": "shop_lookup",
    "confidence_score": 0.9995574355125427
  }
  ```

---
*Generated manually by AI Sales Assistant.*
