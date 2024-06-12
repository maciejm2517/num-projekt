import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from sklearn.metrics import accuracy_score,precision_score,recall_score
import mlflow

DATA_PATH='data/PotatoPlants'
CLASSES=os.listdir(DATA_PATH)
CLASSES.sort()
IMG_WIDTH=300
IMG_HEIGHT=300
RGB=3
NUM_CLASSES=len(CLASSES)
EPOCHS=1

# import data
data=tf.keras.utils.image_dataset_from_directory(DATA_PATH,
                                                 image_size=(IMG_WIDTH,IMG_HEIGHT))

# data augmentation to model
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(IMG_HEIGHT, 
                                                              IMG_WIDTH,
                                                              RGB)),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
])

# scaling data and making batches iterable
data=data.map(lambda x,y: (x/255,y))

# divide data into training, validation and testing
train_size = int(len(data)*0.8)
val_size = int(len(data)*0.1)
test_size = int(len(data)*0.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

# creating model
model=tf.keras.Sequential([
    #data_augmentation,
    tf.keras.layers.Conv2D(16, (3,3),1, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, (3,3),1, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES)    
])

# compiling model
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
               loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), 
               metrics=['accuracy']
               )

with mlflow.start_run():
    history = model.fit(
        train,
        epochs=EPOCHS,
        validation_data=val,
    )
    
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", 32)
    mlflow.log_metric("accuracy", history.history['accuracy'][-1])
    mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1])
    
    mlflow.keras.log_model(model, "model")
