import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.efficientnet import preprocess_input
import pandas as pd
from tensorflow.keras.regularizers import L2

dataset_path = Path(os.getcwd()).parent / "transformed_images"

def get_dataset():
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        label_mode='categorical',
        batch_size=32,
        image_size=(224, 224),
        seed=123,
        validation_split=0.2,
        subset='training'
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        label_mode='categorical',
        batch_size=32,
        image_size=(224, 224),
        seed=123,
        validation_split=0.2,
        subset='validation'
    )

    train_ds = train_ds.map(lambda image, label: (preprocess_input(image), label))
    val_ds = val_ds.map(lambda image, label: (preprocess_input(image), label))

    return train_ds, val_ds


def efficientnetb0_implementation():
    
    NUM_CLASSES = 8
    IMG_SIZE = 224
    
    image_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    effnet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    effnet.trainable = True
    
    x = effnet(image_input)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x) 
    x = layers.Dense(NUM_CLASSES,activation='softmax')(x)
   
    model = tf.keras.Model(inputs=image_input, outputs=x)
    model.summary()
    print("\n","*"*60, "\n")

    return model

def train_model(model):
    
    model.compile(
        optimizer=tf.optimizers.Adam(epsilon=0.0001),
        loss='categorical_crossentropy',
        metrics=["accuracy"]
    )

    chkpnt_loss = tf.keras.callbacks.ModelCheckpoint(
        'best_model_loss.keras',
        monitor='val_loss',         
        verbose=1,                  
        save_best_only=True,        
        mode='min',                 
        save_weights_only=False,    
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=20,
                        callbacks=[chkpnt_loss,early_stopping])


    return history


def plot_hist(hist):

    plt.figure(figsize=(8,3))

    plt.subplot(121)
    hist_df = pd.DataFrame(hist.history)
    hist_df.loc[:,['loss','val_loss']].plot(ax=plt.gca())

    plt.subplot(122)
    hist_df.loc[:,['accuracy','val_accuracy']].plot(ax=plt.gca())

    plt.tight_layout()
    plt.savefig("training_history.png")



if __name__ == "__main__":
    
    # Step 1: Data Preprocessing
    train_ds, val_ds = get_dataset()

    # Step 2: Model Training
    model = efficientnetb0_implementation()
    hist = train_model(model)

    # Step 3: Visualize training
    plot_hist(hist)

    # Step 4: Predict on val dataset
    preds = model.evaluate(val_ds)
    print ("Loss = " + str(preds[0]))
    print ("Validation Accuracy = " + str(preds[1]))