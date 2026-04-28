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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def get_images(directory):
    
    path = Path(os.getcwd()).parent / "transformed_images" / directory
    
    X = []
    y = []
    img_size = (100, 100)
    
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


def train_model(X_valid, y_valid):

    model = Sequential([
        Conv2D(32, (3,3), activation = 'relu', input_shape = (100, 100, 3)),
        MaxPooling2D((2,2)),

        Conv2D(32, (3,3), activation = 'relu'),
        MaxPooling2D((2,2)),

        Flatten(),
        Dense(64, activation = 'relu'),
        Dense(4, activation = 'softmax')])
    

    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.fit(X_train, y_train, epochs = 5, batch_size = 64)

    model.evaluate(X_valid, y_valid)

    return model


def pred_img(model, X_valid, y_valid):
    
    idx2 = random.randint(0, len(y_valid))
    # plt.imshow(X_valid[idx2, :])
    # plt.show()

    y_pred = model.predict(X_valid[idx2, :].reshape(1, 100, 100, 3))
    print(y_pred)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="directory")

    args = parser.parse_args()
    directory = args.d

    X, y = get_images(directory)
    X_train, X_valid, y_train, y_valid = split_dataset(X, y)
    model = train_model(X_valid, y_valid)
    # save_model(model)
    pred_img(model, X_valid, y_valid)


    # Example results

    # Epoch 1/5
    # 163/163 ━━━━━━━━━━━━━━━━━━━━ 10s 59ms/step - accuracy: 0.5477 - loss: 1.0922  
    # Epoch 2/5
    # 163/163 ━━━━━━━━━━━━━━━━━━━━ 10s 61ms/step - accuracy: 0.7628 - loss: 0.6409 
    # Epoch 3/5
    # 163/163 ━━━━━━━━━━━━━━━━━━━━ 10s 62ms/step - accuracy: 0.8396 - loss: 0.4362 
    # Epoch 4/5
    # 163/163 ━━━━━━━━━━━━━━━━━━━━ 11s 68ms/step - accuracy: 0.8734 - loss: 0.3503 
    # Epoch 5/5
    # 163/163 ━━━━━━━━━━━━━━━━━━━━ 11s 69ms/step - accuracy: 0.9064 - loss: 0.2641 
    # 82/82 ━━━━━━━━━━━━━━━━━━━━ 1s 16ms/step - accuracy: 0.7470 - loss: 0.6916
    # [[3.0113186e-03 7.6050006e-02 1.1677919e-07 9.2093861e-01]]

