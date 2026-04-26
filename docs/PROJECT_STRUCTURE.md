# Cấu trúc thư mục

```text
submission_emotion_demo/
  src/
    main.py                    # demo chính, mặc định dùng DeepFace + MTCNN
    emotion_webcam_opencv.py   # bản so sánh dùng OpenCV detector

  data/
    raw/                       # dữ liệu tải về ban đầu, không nộp file lớn
    processed/                 # dữ liệu đã xử lý/chia split, không nộp file lớn

  models/
    .gitkeep                   # nơi đặt model tự train sau khi tải từ Colab

  notebooks/
    .gitkeep                   # nơi đặt notebook Colab nếu nhóm muốn nộp kèm

  reports/
    .gitkeep                   # nơi đặt confusion matrix, training curve, report

  docs/
    PROJECT_STRUCTURE.md       # mô tả cấu trúc thư mục

  assets/
    .gitkeep                   # ảnh minh họa nếu cần

  .deepface/
    weights/
      facial_expression_model_weights.h5  # weights pretrained của DeepFace

  CLAUDE.md
  README.md
  requirements.txt             # dependencies chạy demo local
  requirements-training.txt    # dependencies huấn luyện/evaluate trên Colab
  run.bat                      # chạy demo chính
  run_opencv.bat               # chạy bản OpenCV
```

## Quy ước

- Làm việc chủ yếu trên VS Code trong thư mục này.
- Huấn luyện model trên Google Colab/T4, sau đó tải model về `models/`.
- Không nộp `.venv/`, `.tmp/`, dữ liệu thô lớn trong `data/`, hoặc cache Python.
- `requirements.txt` dùng cho demo/inference local.
- `requirements-training.txt` dùng cho notebook/script huấn luyện.
