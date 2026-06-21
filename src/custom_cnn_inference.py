import json
import os
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "custom_cnn_v1_best.keras"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "models" / "labels.json"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
INPUT_SIZE = (48, 48)


def load_labels(labels_path=DEFAULT_LABELS_PATH):
    labels_path = Path(labels_path)
    labels_data = json.loads(labels_path.read_text(encoding="utf-8"))
    return [labels_data[str(index)] for index in range(len(labels_data))]


def load_emotion_model(model_path=DEFAULT_MODEL_PATH):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    try:
        return tf.keras.models.load_model(model_path)
    except TypeError as error:
        if "quantization_config" not in str(error):
            raise
        model = build_custom_cnn_v1(tf)
        load_weights_from_keras_archive(tf, model, model_path)
        return model


def conv_block(tf, inputs, filters, dropout_rate, block_name):
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        name=f"{block_name}_conv1",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name=f"{block_name}_bn1")(x)
    x = tf.keras.layers.Activation("relu", name=f"{block_name}_relu1")(x)

    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        name=f"{block_name}_conv2",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{block_name}_bn2")(x)
    x = tf.keras.layers.Activation("relu", name=f"{block_name}_relu2")(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name=f"{block_name}_pool")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name=f"{block_name}_dropout")(x)
    return x


def build_custom_cnn_v1(tf):
    inputs = tf.keras.Input(shape=(48, 48, 1), name="input_image")
    x = conv_block(tf, inputs, filters=32, dropout_rate=0.15, block_name="block1")
    x = conv_block(tf, x, filters=64, dropout_rate=0.20, block_name="block2")
    x = conv_block(tf, x, filters=128, dropout_rate=0.25, block_name="block3")
    x = conv_block(tf, x, filters=256, dropout_rate=0.30, block_name="block4")
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(256, use_bias=False, name="classifier_dense")(x)
    x = tf.keras.layers.BatchNormalization(name="classifier_bn")(x)
    x = tf.keras.layers.Activation("relu", name="classifier_relu")(x)
    x = tf.keras.layers.Dropout(0.40, name="classifier_dropout")(x)
    outputs = tf.keras.layers.Dense(7, activation="softmax", name="emotion_probabilities")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="custom_cnn_v1")


def load_weights_from_keras_archive(tf, model, model_path):
    with zipfile.ZipFile(model_path) as archive:
        if "model.weights.h5" not in archive.namelist():
            raise RuntimeError(f"model.weights.h5 not found inside {model_path}")
        with tempfile.NamedTemporaryFile(suffix=".weights.h5") as temp_file:
            temp_file.write(archive.read("model.weights.h5"))
            temp_file.flush()
            model.load_weights(temp_file.name)


def read_image(image_path):
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return image


def preprocess_face(face_bgr):
    if face_bgr.size == 0:
        raise ValueError("Face crop is empty.")

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if face_bgr.ndim == 3 else face_bgr
    resized = cv2.resize(gray, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    return normalized.reshape(1, INPUT_SIZE[1], INPUT_SIZE[0], 1)


def predict_face(model, face_bgr, labels, top_k=3):
    model_input = preprocess_face(face_bgr)
    probabilities = model.predict(model_input, verbose=0)[0]
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    top_predictions = [
        {
            "label_id": int(index),
            "label": labels[int(index)],
            "probability": float(probabilities[index]),
            "percentage": float(probabilities[index] * 100.0),
        }
        for index in top_indices
    ]
    best = top_predictions[0]
    return {
        "predicted_label_id": best["label_id"],
        "predicted_label": best["label"],
        "confidence": best["probability"],
        "confidence_percentage": best["percentage"],
        "top_predictions": top_predictions,
        "probabilities": {
            labels[index]: float(probabilities[index])
            for index in range(len(labels))
        },
    }


def draw_prediction_panel(image_bgr, prediction, origin=(12, 30)):
    output = image_bgr.copy()
    x, y = origin
    cv2.putText(
        output,
        f"{prediction['predicted_label']}: {prediction['confidence_percentage']:.1f}%",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 255),
        2,
    )
    for index, item in enumerate(prediction["top_predictions"], start=1):
        cv2.putText(
            output,
            f"{index}. {item['label']}: {item['percentage']:.1f}%",
            (x, y + 28 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
    return output


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_image(path, image_bgr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image_bgr):
        raise RuntimeError(f"Failed to write image: {path}")
