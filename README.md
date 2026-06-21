# Facial Emotion Recognition - Custom CNN

## 1. Giới thiệu

Dự án xây dựng hệ thống nhận diện cảm xúc khuôn mặt với hai phần chính:

1. Huấn luyện một mô hình **Custom CNN 4 block** trên dataset `abhilash88/fer2013-enhanced`.
2. Tích hợp mô hình đã huấn luyện vào các luồng suy luận thực tế: ảnh crop mặt, ảnh nguyên bản có face detector và webcam theo thời gian thực.

Khác với bản thử nghiệm ban đầu dùng DeepFace pretrained, hướng chính của project là mô hình CNN do nhóm tự thiết kế, tự huấn luyện và tự đánh giá. DeepFace chỉ được giữ lại như một baseline tham khảo để so sánh cảm nhận thực tế.

## 2. Nhãn cảm xúc

Mô hình phân loại ảnh khuôn mặt vào 7 nhãn:

| Nhãn | Ý nghĩa |
|---|---|
| `angry` | Tức giận |
| `disgust` | Ghê tởm / khó chịu |
| `fear` | Sợ hãi |
| `happy` | Vui vẻ |
| `sad` | Buồn |
| `surprise` | Ngạc nhiên |
| `neutral` | Trung tính |

## 3. Pipeline tổng quát

```text
FER2013-enhanced
  -> khảo sát dữ liệu và phân bố lớp
  -> preprocessing ảnh về grayscale 48x48x1
  -> xử lý mất cân bằng bằng sample_weight
  -> augmentation nhẹ trên train set
  -> đóng gói tf.data pipeline
  -> huấn luyện Custom CNN 4 block
  -> đánh giá bằng accuracy, F1, recall, confusion matrix
  -> tích hợp inference ảnh và webcam
```

Khi chạy webcam hoặc ảnh nguyên bản, hệ thống có thêm bước face detector:

```text
Ảnh/webcam frame
  -> Haar hoặc MTCNN phát hiện khuôn mặt
  -> crop vùng mặt
  -> custom_cnn_inference.py tiền xử lý crop về 48x48x1
  -> Custom CNN dự đoán xác suất 7 cảm xúc
  -> vẽ bounding box, nhãn, confidence và FPS
```

## 4. Phân công chính

| Thành viên | MSSV | Phần chính |
|---|---:|---|
| Nguyễn Minh Quân | 20235816 | Trưởng nhóm; thiết kế và huấn luyện Custom CNN; cấu hình optimizer/loss/callbacks; lưu model tốt nhất; báo cáo. |
| Đặng Hoàng Quân | 20235813 | Dataset, khảo sát dữ liệu, preprocessing, xử lý mất cân bằng, augmentation và `tf.data` pipeline. |
| Đinh Văn Phạm Việt | 20235870 | Đánh giá mô hình trên test set, classification report, confusion matrix, confidence và phân tích lỗi sai. |
| Đinh Hữu Nhật Minh | 20235778 | Inference ảnh thực tế, so sánh Haar/MTCNN, tích hợp webcam và kiểm chứng demo. |

## 5. Cấu trúc quan trọng

```text
notebooks/
  train_fer2013_enhanced_colab.ipynb   # notebook chính: dataset -> train -> evaluate

src/
  custom_cnn_inference.py              # lõi load model, preprocess face crop và predict
  infer_image_custom_cnn.py            # suy luận ảnh đã crop mặt sẵn
  infer_face_custom_cnn.py             # suy luận ảnh nguyên bản bằng detector + Custom CNN
  face_detectors.py                    # Haar Cascade và MTCNN detector
  compare_face_detectors.py            # so sánh Haar/MTCNN trên folder ảnh
  main_custom_cnn.py                   # webcam realtime bằng Custom CNN
  main.py                              # webcam baseline dùng DeepFace pretrained
  list_cameras.py                      # dò camera index trên máy local

models/
  custom_cnn_v1_best.keras             # checkpoint tốt nhất từ Colab
  labels.json                          # mapping id -> emotion label

reports/
  *.json, *.csv, *.png                 # kết quả khảo sát, train, evaluate, detector và inference
```

## 6. Môi trường

Project được phát triển trên Python 3.10+.

Cài môi trường local:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Nếu chạy trên Windows:

```bat
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

Lưu ý:

- Huấn luyện nên chạy trên Google Colab T4 bằng notebook trong `notebooks/`.
- Local chủ yếu dùng để chạy inference ảnh, so sánh detector và webcam.
- Để chạy Custom CNN local, cần có `models/custom_cnn_v1_best.keras` và `models/labels.json`.

## 7. Huấn luyện và đánh giá trên Colab

Notebook chính:

```text
notebooks/train_fer2013_enhanced_colab.ipynb
```

Notebook này gồm các bước:

1. Tải dataset `abhilash88/fer2013-enhanced`.
2. Khảo sát split, label mapping, phân bố lớp và chất lượng ảnh.
3. Tiền xử lý ảnh thành tensor `48x48x1`.
4. Xử lý mất cân bằng bằng `sample_weight` và tạo augmentation nhẹ.
5. Tạo `tf.data.Dataset` cho train/validation/test.
6. Xây Custom CNN 4 block.
7. Train model với AdamW, sparse categorical crossentropy, ModelCheckpoint, EarlyStopping và ReduceLROnPlateau.
8. Đánh giá test set bằng accuracy, balanced accuracy, precision, recall, F1-score, top-2 accuracy và confusion matrix.

Sau khi train xong trên Colab, tải các file quan trọng về local:

```text
models/custom_cnn_v1_best.keras
models/labels.json
reports/custom_cnn_v1_*.json
reports/custom_cnn_v1_*.csv
reports/custom_cnn_v1_*.png
```

## 8. Chạy webcam bằng Custom CNN

Đây là lệnh webcam chính của project:

```bash
.venv/bin/python src/main_custom_cnn.py --source 0
```

Trên Windows:

```bat
.venv\Scripts\python src\main_custom_cnn.py --source 0
```

Mặc định script dùng:

```text
--detector mtcnn
--every 8
--smooth 8
--padding 0.18
--top-k 3
```

Ý nghĩa:

- `--source 0`: dùng camera index 0.
- `--detector mtcnn`: dùng MTCNN để tìm khuôn mặt trước khi đưa vào Custom CNN.
- `--every 8`: chỉ phân tích mỗi 8 frame để giảm lag.
- `--smooth 8`: lấy trung bình xác suất của 8 lần dự đoán gần nhất để kết quả đỡ nhảy.
- `--padding 0.18`: mở rộng bounding box quanh mặt để không cắt mất vùng trán/cằm.
- `--top-k 3`: hiển thị 3 cảm xúc có xác suất cao nhất.

Thoát cửa sổ webcam bằng phím `q`.

Nếu camera mặc định không đúng, dò camera bằng:

```bash
.venv/bin/python src/list_cameras.py --max-index 5
```

Sau đó thử camera khác:

```bash
.venv/bin/python src/main_custom_cnn.py --source 1
```

Nếu muốn dùng Haar Cascade thay vì MTCNN:

```bash
.venv/bin/python src/main_custom_cnn.py --source 0 --detector haar
```

Nếu muốn lưu log từng lần phân tích webcam:

```bash
.venv/bin/python src/main_custom_cnn.py --source 0 --save-log
```

Log sẽ được lưu vào:

```text
reports/webcam_logs/
```

## 9. Chạy inference trên ảnh

### 9.1. Ảnh đã crop mặt sẵn

Dùng khi ảnh đầu vào chỉ chứa vùng mặt:

```bash
.venv/bin/python src/infer_image_custom_cnn.py \
  --image image/crop_hppy1.png \
  --output-image reports/crop_hppy1_result.png \
  --output-json reports/crop_hppy1_result.json
```

Script này không chạy face detector. Nó đưa ảnh crop trực tiếp qua `custom_cnn_inference.py` để preprocess và predict.

### 9.2. Ảnh nguyên bản có nền

Dùng khi ảnh đầu vào là ảnh bình thường, chưa crop mặt:

```bash
.venv/bin/python src/infer_face_custom_cnn.py \
  --image image/full_hppy1.png \
  --detector mtcnn \
  --output-image reports/full_hppy1_result.png \
  --output-json reports/full_hppy1_result.json
```

Có thể đổi sang Haar:

```bash
.venv/bin/python src/infer_face_custom_cnn.py \
  --image image/full_hppy1.png \
  --detector haar
```

Luồng xử lý của script này:

```text
ảnh nguyên bản -> face detector -> crop mặt -> preprocess 48x48x1 -> Custom CNN predict
```

## 10. So sánh Haar và MTCNN

Để tạo kết quả so sánh detector trên các ảnh trong folder `image/`:

```bash
.venv/bin/python src/compare_face_detectors.py --image-dir image
```

Kết quả được lưu vào:

```text
reports/detector_comparison/
  detector_comparison.csv
  detector_comparison.json
  detector_comparison_summary.json
  haar_*.png
  mtcnn_*.png
```

Ý nghĩa báo cáo:

- Haar Cascade thường nhanh hơn nhưng nhạy với ánh sáng, góc mặt và chất lượng ảnh.
- MTCNN chậm hơn nhưng ổn định hơn trong nhiều ảnh thực tế.
- Custom CNN chỉ phân loại cảm xúc sau khi đã có vùng mặt; vì vậy chất lượng detector ảnh hưởng trực tiếp tới kết quả cuối.

## 11. DeepFace baseline

File [src/main.py](src/main.py) là bản webcam dùng DeepFace pretrained. Đây không phải hướng chính của nhóm, nhưng được giữ lại để đối chiếu:

```bash
.venv/bin/python src/main.py --mode fast --source 0
```

Nên hiểu phần này là baseline tham khảo:

- DeepFace dùng mô hình pretrained, có thể cho kết quả thực tế tốt hơn trong một số ảnh.
- Custom CNN là phần đóng góp chính vì nhóm tự xử lý dữ liệu, tự thiết kế, tự huấn luyện và tự đánh giá.

## 12. Vai trò của `custom_cnn_inference.py`

`src/custom_cnn_inference.py` là module trung gian giữa model đã train và các script chạy thực tế. File này không train lại model. Nhiệm vụ của nó là:

- Load `models/custom_cnn_v1_best.keras`.
- Load `models/labels.json`.
- Dựng lại kiến trúc `custom_cnn_v1` khi cần để tránh lỗi khác phiên bản TensorFlow/Keras.
- Chuyển ảnh mặt sang grayscale.
- Resize về `48x48`.
- Normalize pixel về `[0, 1]`.
- Reshape thành batch `(1, 48, 48, 1)`.
- Chạy `model.predict`.
- Trả về top-k cảm xúc và xác suất của đủ 7 lớp.

Các file `infer_image_custom_cnn.py`, `infer_face_custom_cnn.py`, `compare_face_detectors.py` và `main_custom_cnn.py` đều dùng lại module này.

## 13. Kết quả chính

Mô hình Custom CNN 4 block đạt kết quả test xấp xỉ:

| Chỉ số | Giá trị |
|---|---:|
| Test accuracy | 62.65% |
| Balanced accuracy | 62.07% |
| Macro F1-score | 59.09% |
| Weighted F1-score | 62.19% |
| Top-2 accuracy | 81.60% |

Kết quả này phù hợp với độ khó của FER2013-enhanced: ảnh nhỏ `48x48`, grayscale, nhiều nhãn cảm xúc gần nhau và mất cân bằng lớp mạnh, đặc biệt lớp `disgust` có ít mẫu hơn nhiều so với `happy`.

## 14. Hạn chế

- Dataset huấn luyện là ảnh grayscale 48x48, trong khi ảnh webcam thực tế là ảnh màu, kích thước lớn và chịu ảnh hưởng ánh sáng.
- Các cảm xúc như `fear`, `sad`, `disgust` dễ nhầm với `neutral`, `angry` hoặc `surprise`.
- Face detector và emotion classifier là hai bước riêng; detector crop lệch sẽ làm classifier dự đoán kém.
- MTCNN ổn định hơn Haar nhưng chậm hơn.
- Webcam thực tế có thể bị nhiễu, cháy sáng, thiếu sáng hoặc lệch góc mặt.

## 15. Tài liệu tham khảo

- FER2013-enhanced dataset: `abhilash88/fer2013-enhanced`
- TensorFlow / Keras: https://www.tensorflow.org/
- OpenCV: https://opencv.org/
- MTCNN: Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks
- DeepFace: https://github.com/serengil/deepface
