# Hệ thống nhận diện cảm xúc khuôn mặt theo thời gian thực

## 1. Giới thiệu đề tài

Dự án xây dựng một hệ thống nhận diện cảm xúc khuôn mặt theo thời gian thực bằng webcam. Hệ thống sử dụng các kỹ thuật thị giác máy tính và mô hình học sâu đã huấn luyện sẵn để:

- Đọc luồng hình ảnh trực tiếp từ webcam.
- Phát hiện khuôn mặt trong từng khung hình.
- Phân loại cảm xúc khuôn mặt.
- Hiển thị bounding box, nhãn cảm xúc, xác suất cảm xúc và FPS lên màn hình.

Đây là một ví dụ ứng dụng trí tuệ nhân tạo trong bài toán **Facial Emotion Recognition**. Hệ thống không tự huấn luyện mô hình từ đầu, mà tập trung vào việc tích hợp mô hình pretrained, xây dựng pipeline xử lý ảnh thời gian thực và đánh giá khả năng hoạt động trong môi trường webcam thực tế.

## 2. Các cảm xúc được nhận diện

Mô hình phân loại cảm xúc theo 7 nhãn:

| Nhãn | Ý nghĩa |
|---|---|
| `angry` | Tức giận |
| `disgust` | Ghê tởm / khó chịu |
| `fear` | Sợ hãi |
| `happy` | Vui vẻ |
| `sad` | Buồn |
| `surprise` | Ngạc nhiên |
| `neutral` | Trung tính |

Kết quả cuối cùng được chọn theo cảm xúc có xác suất cao nhất sau bước làm mượt.

## 3. Công nghệ sử dụng

| Thành phần | Vai trò |
|---|---|
| Python | Ngôn ngữ lập trình chính |
| OpenCV | Đọc webcam, xử lý frame, vẽ bounding box và text |
| DeepFace | Thư viện AI cung cấp mô hình nhận diện cảm xúc pretrained |
| TensorFlow / Keras | Backend chạy mô hình học sâu |
| MTCNN | Mô hình phát hiện khuôn mặt mặc định |
| Haar Cascade OpenCV | Detector nhẹ hơn, dùng để so sánh với MTCNN |

## 4. Pipeline xử lý

Pipeline tổng quát:

```text
Webcam
  -> OpenCV đọc frame
  -> phát hiện khuôn mặt
  -> cắt và tiền xử lý vùng mặt
  -> mô hình CNN phân loại cảm xúc
  -> lấy xác suất 7 cảm xúc
  -> làm mượt kết quả qua nhiều frame
  -> vẽ bounding box, nhãn cảm xúc và FPS
  -> hiển thị video realtime
```

Chi tiết từng bước:

1. **Đọc frame từ webcam**

   OpenCV sử dụng `cv2.VideoCapture(0)` để mở camera mặc định. Mỗi vòng lặp, chương trình đọc một frame ảnh từ webcam.

2. **Phát hiện khuôn mặt**

   Bản mặc định sử dụng MTCNN thông qua DeepFace:

   ```text
   DeepFace.analyze(..., detector_backend="mtcnn")
   ```

   MTCNN phát hiện vị trí khuôn mặt và trả về bounding box gồm tọa độ `x`, `y`, `w`, `h`.

3. **Phân loại cảm xúc**

   Sau khi có vùng khuôn mặt, DeepFace đưa ảnh mặt vào mô hình emotion pretrained. Mô hình trả về xác suất cho 7 lớp cảm xúc.

4. **Làm mượt kết quả**

   Dự đoán cảm xúc trên webcam thường bị nhảy giữa các frame. Vì vậy chương trình lưu lại một số kết quả gần nhất và lấy trung bình xác suất.

   Tham số mặc định:

   ```text
   --smooth 8
   ```

   Nghĩa là lấy trung bình 8 lần dự đoán gần nhất.

5. **Tối ưu realtime**

   Việc chạy model trên mọi frame sẽ gây lag. Chương trình chỉ phân tích mỗi `N` frame:

   ```text
   --every 12
   ```

   Ngoài ra, phần phân tích AI được chạy trong background thread bằng `ThreadPoolExecutor`, giúp webcam vẫn hiển thị mượt trong khi model đang xử lý.

6. **Hiển thị kết quả**

   Chương trình vẽ:

   - Bounding box quanh khuôn mặt.
   - Nhãn cảm xúc chính.
   - Xác suất cảm xúc.
   - FPS.
   - Trạng thái `Analyzing...` hoặc `No face detected`.

## 5. Hai chế độ detector

### 5.1. Chế độ mặc định: MTCNN

Chạy bằng:

```bat
run.bat
```

Mặc định tương đương:

```bat
python src\main.py --mode deep --detector mtcnn
```

Ưu điểm:

- Phát hiện khuôn mặt tốt hơn trong điều kiện thực tế.
- Ổn hơn khi người dùng đeo kính, mặt hơi nghiêng hoặc ánh sáng không hoàn hảo.

Nhược điểm:

- Chậm hơn OpenCV Haar Cascade.
- Cần tối ưu bằng `--every`, `--smooth` và background thread.

### 5.2. Chế độ so sánh: OpenCV

Chạy bằng:

```bat
run_opencv.bat
```

Tương đương:

```bat
python src\emotion_webcam_opencv.py
```

Ưu điểm:

- Nhẹ hơn.
- Có thể nhanh hơn trên máy yếu.

Nhược điểm:

- Dễ không phát hiện được mặt khi ánh sáng yếu, mặt bị nghiêng hoặc đeo kính.
- Độ ổn định thấp hơn MTCNN trong thử nghiệm thực tế.

## 6. Cấu trúc thư mục

```text
submission_emotion_demo/
  src/
    main.py                    # chương trình chính, mặc định dùng MTCNN
    emotion_webcam_opencv.py   # bản so sánh dùng OpenCV detector
  docs/
    PROJECT_STRUCTURE.md       # mô tả cấu trúc project
  assets/
    .gitkeep                   # thư mục để ảnh minh họa nếu cần
  .deepface/
    weights/
      facial_expression_model_weights.h5
  requirements.txt             # danh sách thư viện cần cài
  run.bat                      # chạy bản chính
  run_opencv.bat               # chạy bản OpenCV
  README.md                    # tài liệu hướng dẫn
```

## 7. Cài đặt môi trường

Yêu cầu:

- Windows
- Python 3.10 trở lên
- Webcam
- Internet khi cài thư viện lần đầu

Tạo môi trường ảo:

```bat
python -m venv .venv
```

Kích hoạt môi trường:

```bat
.venv\Scripts\activate
```

Cài thư viện:

```bat
python -m pip install -r requirements.txt
```

Sau khi cài xong, có thể chạy demo bằng file `.bat`.

## 8. Cách chạy

Chạy bản chính dùng MTCNN:

```bat
run.bat
```

Chạy bản so sánh dùng OpenCV:

```bat
run_opencv.bat
```

Thoát cửa sổ webcam bằng phím:

```text
q
```

## 9. Các tham số tùy chỉnh

### Chọn camera khác

Nếu máy có nhiều camera:

```bat
run.bat --source 1
```

Camera mặc định là:

```text
--source 0
```

### Giảm lag

Tăng số frame bỏ qua giữa các lần phân tích:

```bat
run.bat --every 20
```

Giá trị càng lớn thì chương trình càng nhẹ, nhưng cảm xúc cập nhật chậm hơn.

### Phản hồi nhanh hơn

```bat
run.bat --every 6
```

Giá trị nhỏ giúp cảm xúc cập nhật nhanh hơn, nhưng có thể gây giật hơn.

### Làm mượt kết quả hơn

```bat
run.bat --smooth 12
```

Giá trị `smooth` càng lớn thì kết quả ổn định hơn, nhưng phản ứng với thay đổi cảm xúc chậm hơn.

### Chạy OpenCV detector từ file chính

```bat
run.bat --mode deep --detector opencv
```

### Chạy mode fast

```bat
run.bat --mode fast
```

Mode này dùng OpenCV Haar Cascade để phát hiện mặt, sau đó chỉ đưa vùng mặt vào mô hình emotion. Trên một số điều kiện webcam, mode này có thể nhanh nhưng dễ hụt mặt hơn MTCNN.

## 10. Mô hình sử dụng

### 10.1. MTCNN

MTCNN là mô hình phát hiện khuôn mặt theo kiến trúc cascade gồm 3 mạng:

- **P-Net**: tạo các vùng ứng viên có thể chứa khuôn mặt.
- **R-Net**: lọc và tinh chỉnh bounding box.
- **O-Net**: xác nhận khuôn mặt và dự đoán landmark.

Trong project này, MTCNN được sử dụng để tìm vị trí khuôn mặt trong frame webcam.

### 10.2. DeepFace Emotion Model

DeepFace cung cấp mô hình phân loại cảm xúc pretrained. Ảnh khuôn mặt được tiền xử lý và đưa vào mô hình CNN. Mô hình trả về xác suất cho 7 cảm xúc.

File weights được đặt tại:

```text
.deepface/weights/facial_expression_model_weights.h5
```

Việc để sẵn file weights giúp demo không cần tải model lại khi chạy lần đầu.

## 11. Đánh giá theo PEAS

| Thành phần | Mô tả |
|---|---|
| Performance Measure | Độ chính xác phát hiện khuôn mặt, độ hợp lý của nhãn cảm xúc, FPS, độ trễ cập nhật cảm xúc |
| Environment | Người dùng ngồi trước webcam trong phòng học hoặc phòng cá nhân, ánh sáng có thể thay đổi |
| Actuators | Hiển thị bounding box, nhãn cảm xúc, xác suất, FPS và trạng thái lên cửa sổ video |
| Sensors | Webcam, frame ảnh từ OpenCV, detector khuôn mặt, mô hình emotion |

## 12. Hạn chế

Hệ thống còn một số hạn chế:

- Kết quả cảm xúc phụ thuộc mạnh vào ánh sáng.
- Webcam bị cháy sáng hoặc thiếu sáng có thể làm model dự đoán sai.
- Kính, tóc che mắt, góc mặt nghiêng có thể ảnh hưởng tới cả detector và emotion model.
- Mô hình emotion pretrained không đảm bảo chính xác tuyệt đối trong mọi trường hợp.
- Một số cảm xúc gần nhau như `happy`, `surprise`, `fear` có thể bị nhầm lẫn.
- Hệ thống hiện chỉ phục vụ demo realtime, chưa có chức năng lưu log hoặc thống kê kết quả.

## 13. Hướng phát triển

Một số hướng có thể cải thiện:

- Thu thập dữ liệu cảm xúc riêng và fine-tune mô hình.
- Thử các detector mạnh hơn như RetinaFace hoặc YOLO face detector.
- Thêm chức năng lưu kết quả theo thời gian.
- Thêm giao diện người dùng thay vì chỉ dùng cửa sổ OpenCV.
- Đánh giá định lượng trên tập ảnh có nhãn thay vì chỉ quan sát trực tiếp qua webcam.

## 14. Tài liệu tham khảo

- DeepFace: https://github.com/serengil/deepface
- OpenCV: https://opencv.org/
- TensorFlow: https://www.tensorflow.org/
- MTCNN paper: Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks
