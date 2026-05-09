import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys   # noqa: E402
import cv2  # noqa: E402
from tensorflow.keras.models import load_model   # noqa: E402
from tensorflow.keras.applications.efficientnet \
    import preprocess_input  # noqa: E402
import numpy as np  # noqa: E402
from pathlib import Path  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSFORMATION = PROJECT_ROOT / "03_image_transformation"
MODEL_PATH = PROJECT_ROOT / "04_classification" / "model.keras"

sys.path.insert(0, str(TRANSFORMATION))
os.chdir(TRANSFORMATION)

from transformation import img_transformation  # noqa: E402

CLASSES = ['Apple_Black_rot', 'Apple_healthy', 'Apple_rust', 'Apple_scab',
           'Grape_Black_rot', 'Grape_Esca', 'Grape_healthy', 'Grape_spot']


def predict(img, model):

    predictions = model.predict(img, verbose=0)

    class_id = (np.argmax(predictions[0]))
    confidence = np.max(predictions[0])

    predicted_class = CLASSES[class_id]
    return predicted_class, confidence


def preprocess_image(image_path):

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    return img


def display_images(img, leaf_mask):

    img_display = img[0]

    plt.figure(figsize=(12, 4))

    plt.subplot(121)
    plt.imshow(img_display)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(leaf_mask, cmap="gray")
    plt.title("Transformed Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # python3 predict.py "/home/eablak/Desktop/leaffliction
    # /augmented_directory/images/Apple_healthy/image (2).JPG"

    image_path = sys.argv[1]
    model = load_model(MODEL_PATH)

    img = preprocess_image(image_path)
    leaf_mask = img_transformation(image_path, "mask")

    predicted_class, confidence = predict(img, model)
    display_images(img, leaf_mask)
    print("Predicted class is: ", predicted_class,
          " with ", confidence, " confidence")
