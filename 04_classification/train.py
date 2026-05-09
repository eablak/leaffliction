import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from tensorflow.keras.applications import EfficientNetB0  # noqa: E402
from tensorflow.keras import layers  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.keras.applications.efficientnet \
    import preprocess_input  # noqa: E402
import pandas as pd  # noqa: E402
import sys  # noqa: E402
import zipfile  # noqa: E402
matplotlib.use('Agg')  # Use non-GUI backend


def get_dataset(dataset_path):

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

    train_ds = train_ds.map(lambda image,
                            label: (preprocess_input(image), label))
    val_ds = val_ds.map(lambda image, label: (preprocess_input(image), label))

    return train_ds, val_ds


def efficientnetb0_implementation():

    NUM_CLASSES = 8
    IMG_SIZE = 224

    image_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    effnet = EfficientNetB0(weights='imagenet', include_top=False,
                            input_shape=(IMG_SIZE, IMG_SIZE, 3))
    effnet.trainable = True

    x = effnet(image_input)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = tf.keras.Model(inputs=image_input, outputs=x)
    model.summary()
    print("\n", "*"*60, "\n")

    return model


def train_model(model, train_ds, val_ds):

    model.compile(
        optimizer=tf.optimizers.Adam(epsilon=0.0001),
        loss='categorical_crossentropy',
        metrics=["accuracy"]
    )

    chkpnt_loss = tf.keras.callbacks.ModelCheckpoint(
        'model.keras',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        save_weights_only=False,
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=20,
                        callbacks=[chkpnt_loss, early_stopping])

    return history


def plot_hist(hist):

    plt.figure(figsize=(8, 3))

    plt.subplot(121)
    hist_df = pd.DataFrame(hist.history)
    hist_df.loc[:, ['loss', 'val_loss']].plot(ax=plt.gca())

    plt.subplot(122)
    hist_df.loc[:, ['accuracy', 'val_accuracy']].plot(ax=plt.gca())

    plt.tight_layout()
    plt.savefig("training_history.png")


def save_in_zip():

    with zipfile.ZipFile("final.zip", "w") as zipf:

        zipf.write("model.keras")
        zipf.write("train.txt")
        zipf.write("training_history.png")

        for root, dirs, files in os.walk("../augmented_directory"):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, "../augmented_directory")
                zipf.write(file_path, arcname)


if __name__ == "__main__":

    # python3 train.py "full path of augmented_directory/images"
    dataset_path = sys.argv[1]

    # Step 1: Data Preprocessing
    train_ds, val_ds = get_dataset(dataset_path)

    # Step 2: Model Training
    model = efficientnetb0_implementation()
    hist = train_model(model, train_ds, val_ds)

    # Step 3: Visualize training
    plot_hist(hist)

    # Step 4: Predict on val dataset
    preds = model.evaluate(val_ds)
    print("Loss = " + str(preds[0]))
    print("Validation Accuracy = " + str(preds[1]))

    # Step 5: Create a zip file
    save_in_zip()
