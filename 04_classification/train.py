import cv2
import numpy as np
import random
import argparse
from pathlib import Path
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.regularizers import L2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping

def get_images(directory):
    
    path = Path(os.getcwd()).parent / "transformed_images" / directory
    
    X = []
    y = []
    img_size = (128, 128)
    
    for folder in os.listdir(path):
        path_folder = str(path) + "/" + folder

        for img_name in os.listdir(Path(path_folder)):

            img = cv2.imread(path_folder + "/" + img_name)
            img = cv2.resize(img, img_size)
            # print(img.shape)
            X.append(img)
            y.append(folder)

    
    X = np.array(X)
    y = np.array(y)

    X = X / 255.0
    y = y.reshape(len(y), 1)
    y = LabelEncoder().fit_transform(y.ravel())

    # print(X.shape)
    # print(y.shape)

    return X, y


def split_dataset(X, y):
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y,test_size=0.2,train_size=0.8)
    
    # print(X_train.shape)
    # print(X_valid.shape)
    # print(y_train.shape)
    # print(y_train.shape)

    return X_train, X_valid, y_train, y_valid


def train_model(X_train, X_valid, y_train, y_valid):

    model = Sequential([
        Conv2D(32, (3,3), activation = 'relu', input_shape = (128, 128, 3)),
        MaxPooling2D((2,2)),


        Conv2D(64, (3,3), activation = 'relu'),
        MaxPooling2D((2,2)),
        Dropout(0.15),


        Conv2D(128, (3,3), activation = 'relu'),
        MaxPooling2D((2,2)),


        Flatten(),
        Dense(128, activation = 'relu', kernel_regularizer=L2(0.0001)),
        Dropout(0.35),
        Dense(4, activation = 'softmax')
    ])
    

    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    callbacks = [EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, mode='max')]

    model.fit(X_train, y_train, epochs = 20, batch_size = 64, validation_data=(X_valid, y_valid), callbacks=callbacks)

    model.evaluate(X_valid, y_valid)

    return model


def save_model(model, directory):

    file_name = directory + "_model.h5"
    model.save(file_name, include_optimizer=True)
    print("Model saved!")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="directory")

    args = parser.parse_args()
    directory = args.d

    X, y = get_images(directory)
    X_train, X_valid, y_train, y_valid = split_dataset(X, y)
    
    model = train_model(X_train, X_valid, y_train, y_valid)
    model.summary()

    save_model(model, directory)
