import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
import sys
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np

CLASSES = [ 'Apple_Black_rot', 'Apple_healthy', 'Apple_rust', 'Apple_scab', 'Grape_Black_rot', 'Grape_Esca', 'Grape_healthy', 'Grape_spot']


def predict(img, model):

    predictions = model.predict(img, verbose=0)

    class_id = (np.argmax(predictions[0]))
    confidence = np.max(predictions[0])

    predicted_class = CLASSES[class_id]

    print("Predicted class is: ", predicted_class, " with ", confidence, " confidence")


def preprocess_image(image_path):
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    return img


if __name__ == "__main__":

    # python3 predict.py "/home/eablak/Desktop/leaffliction/transformed_images/Grape_spot/image (2)_mask.JPG"

    image_path = sys.argv[1]
    model = load_model('best_model_loss.keras')

    img = preprocess_image(image_path)

    predict(img, model)