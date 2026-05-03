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

    return df


def get_images():

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

    images = np.array(images)
    return images
    

def resize_images(images):

    images = images.astype('float32') / 255.0

    # images.shape
    return images


def label_encoding(df):

    y = df["Labels"].values

    le = LabelEncoder()
    y = le.fit_transform(y)

    print(y)
    print("\n","*"*60)

    return y

def y_one_hot(y):

    y = y.reshape(-1,1)

    ct = ColumnTransformer([('my_ohe', OneHotEncoder(), [0])], remainder='passthrough')
    Y = ct.fit_transform(y)

    print(Y[:5])
    print("\n","*"*60, "\n")

    return Y


def split_dataset(images, Y):

    images, Y = shuffle(images, Y, random_state=1)

    x_train, x_test, y_train, y_test = train_test_split(images, Y, test_size=0.2, random_state=415)

    #inpect the shape of the training and testing.
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    print("\n","*"*60, "\n")

    return x_train, x_test, y_train, y_test


def efficientnetb0_implementation():
    
    NUM_CLASSES = 8
    IMG_SIZE = 224
    
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Using model without transfer learning
    outputs = EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)(inputs)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"] )

    model.summary()
    print("\n","*"*60, "\n")

    return model


def train_model(model, x_train, y_train):
    
    hist = model.fit(x_train, y_train, epochs=30, verbose=2)
    




if __name__ == "__main__":
    
    # Step 1: Data Preprocessing

    get_class_infos()
    df = convert_images_to_df()
    
    images = get_images()
    # images = resize_images(images) # call this while training

    y = label_encoding(df)
    Y = y_one_hot(y)

    x_train, x_test, y_train, y_test = split_dataset(images, Y)


    # Step 2: Model Training

    model = efficientnetb0_implementation()
    train_model(model, x_train, y_train)