# pip install "kagglehub[pandas-datasets]"
#from tensorflow import keras
#import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
# from sklearn.model_selection import train_test_split

# configuring the dataset
'''import os
import shutil

source_dir = "dataset"
target_dir = "split_dataset"

classes = ["Retro", "Typography-Focused"]

for cls in classes:

    class_path = os.path.join(source_dir, cls)
    images = [f for f in os.listdir(class_path) if f.endswith((".jpg",".png",".jpeg"))]

    # Train / temp split
    train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)

    # Temp -> val + test
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    splits = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs
    }

    for split, img_list in splits.items():

        split_folder = os.path.join(target_dir, split, cls)
        os.makedirs(split_folder, exist_ok=True)

        for img in img_list:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_folder, img)
            shutil.copy(src, dst)

print("Dataset split complete.")'''

# resize the data
train_ds = tf.keras.utils.image_dataset_from_directory(
    "split_dataset/train",
    image_size=(224,224),
    batch_size=32,
    color_mode="rgb"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "split_dataset/val",
    image_size=(224,224),
    batch_size=32,
    color_mode="rgb"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "split_dataset/test",
    image_size=(224,224),
    batch_size=32,
    color_mode="rgb"
)

# data augmentation
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
])
train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))

# Normalize images
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

model = keras.Sequential([
    keras.Input(shape=(224,224,3)),

    keras.layers.Conv2D(32,3,activation="relu"),
    keras.layers.MaxPooling2D(),

    keras.layers.Conv2D(64,3,activation="relu"),
    keras.layers.MaxPooling2D(),

    keras.layers.Flatten(),

    keras.layers.Dense(128,activation="relu"),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(1,activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_ds, epochs=15, validation_data=val_ds)

test_loss, test_acc = model.evaluate(test_ds)
print("Test accuracy:", test_acc)

model.save("book_model.keras")
