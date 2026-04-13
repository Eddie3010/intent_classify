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
# Phân loại Intent (Intent Descriptions)

Tài liệu này mô tả chi tiết 6 nhãn ý định (intent labels) chính được sử dụng trong tập dữ liệu huấn luyện và dự đoán của hệ thống **AI Sales Assistant**. Mỗi intent đại diện cho một mục đích cụ thể trong luồng giao tiếp của người dùng.

---

### 1. `product_search` (Tìm kiếm sản phẩm)
- **Mô tả:** Khách hàng muốn tìm xem hệ thống (cửa hàng) có bán hoặc kinh doanh một dòng sản phẩm điện thoại / máy tính bảng cụ thể nào đó hay không.
- **Ví dụ câu hỏi (`current_query`):**
  - *"Shop có Xiaomi 13 hông?"*
  - *"Cho chú hỏi có bán Samsung Galaxy S24 không?"*
  - *"Bên mình còn iPhone 14 không em?"*

---

### 2. `product_detail` (Chi tiết & Thông số kỹ thuật)
- **Mô tả:** Khách hàng thắc mắc về thông số cấu hình, độ bền, trải nghiệm thời lượng pin hoặc tình trạng tỏa nhiệt khi sử dụng một thiết bị di động.
- **Ví dụ câu hỏi (`current_query`):**
  - *"Máy này pin có trâu không cháu?"*
  - *"Dùng lâu có nóng máy không?"*
  - *"Cấu hình chơi game ổn không ạ?"*

---

### 3. `product_compare` (So sánh sản phẩm)
- **Mô tả:** Người dùng phân vân giữa hai (hoặc nhiều) phiên bản, hãng sản xuất khác nhau và yêu cầu bot tư vấn so sánh giữa thiết bị họ đang quan tâm với một đối thủ khác.
- **Ví dụ câu hỏi (`current_query`):**
  - *"So giúp anh với Samsung được không?"*
  - *"So với Xiaomi 13 thì cái nào bền hơn?"*
  - *"Chơi game nên chọn máy nào giữa hai con này?"*

---

### 4. `product_inventory` (Tồn kho sản phẩm tại vị trí)
- **Mô tả:** Ý định khách hàng muốn kiểm tra xem một loại sản phẩm có còn hàng sẵn ở một địa điểm/huyện/tỉnh thành cụ thể hay là gần nhà khách hay không.
- **Ví dụ câu hỏi (`current_query`):**
  - *"Gần nhà cô có sẵn không con?"*
  - *"Ở Bình Dương còn hàng không em?"*
  - *"Bình Dương còn máy này không?"*

---

### 5. `product_promotion` (Khuyến mãi & Thanh toán)
- **Mô tả:** Khách hàng quan tâm đến chính sách giá, ưu đãi, giảm giá thanh toán qua các ngân hàng đối tác hoặc khuyến mãi dành cho các tệp khách đặc biệt (HSSV).
- **Ví dụ câu hỏi (`current_query`):**
  - *"Hiện máy này có khuyến mãi gì không?"*
  - *"Có ưu đãi gì cho học sinh sinh viên không?"*
  - *"Thanh toán qua Sacombank có giảm không em?"*

---

### 6. `shop_lookup` (Tìm kiếm thông tin cửa hàng)
- **Mô tả:** Khách hàng muốn biết địa chỉ cụ thể của cơ sở đang hoạt động, yêu cầu định vị cửa hàng nằm trong một quận/huyện/tỉnh nào đó.
- **Ví dụ câu hỏi (`current_query`):**
  - *"Chỉ chú shop gần đây với"*
  - *"Nhà ở Thủ Đức thì mua ở đâu gần nhất?"*
  - *"Có shop nào gần Biên Hòa Đồng Nai không?"*

