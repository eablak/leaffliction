import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import cv2
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_path = Path(os.getcwd()).parent / "transformed_images"


def get_class_infos():

    print("Dataset classes are:", [x for x in os.listdir(dataset_path)])
    print("\nTypes of classes labels:", len(os.listdir(dataset_path)))
    print("\n", "*"*60, "\n")


def convert_images_to_df():
    
    data_list = []

    for root, dirs, files in os.walk(dataset_path):
        class_name = os.path.basename(root)
        if class_name == "transformed_images":
            continue
        for file in files:
            data_list.append([class_name, os.path.join(root, file)])

    df = pd.DataFrame(data=data_list, columns=['Labels', 'image'])

    print(df.head())
    print(df.tail())
    print("\n")

    label_count = df['Labels'].value_counts()
    print(label_count)
    print("\n","*"*60)


def resize_images():

    im_size = 224

    images = []
    labels = []

    for data_path in dataset_path.iterdir():
        filenames = [i for i in os.listdir(data_path) ]
    
        for f in filenames:
            img = cv2.imread(str(data_path) + '/' + f)
            img = cv2.resize(img, (im_size, im_size))
            images.append(img)
            labels.append(data_path.name)

    datagen = ImageDataGenerator(rescale = 1./255)
    
    train_generator = datagen.flow_from_directory(str(dataset_path), target_size=(224, 224), batch_size=32)
    images_batch, labels_batch = next(train_generator)

    # rn per batch. write it into func and call it in model training time
    print(images_batch.shape)
    print(labels_batch.shape)


if __name__ == "__main__":
    
    # Step 1: Data Preprocessing

    get_class_infos()
    convert_images_to_df()
    resize_images()