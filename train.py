import typer
import tensorflow as tf
import os
import numpy as np
from typing_extensions import Annotated
import mlflow

def main(batch_size: int = typer.Option(32, "--batch_size", "-b", help="Batch size for training"),
         learning_rate: float = typer.Option(1e-4, "--learning_rate", "-lr", help="Learning rate for optimizer"),
         epochs: int = typer.Option(1, "--epochs", "-e", help="Number of epochs")):
    DATA_PATH='data/PotatoPlants'
    CLASSES=os.listdir(DATA_PATH)
    IMG_WIDTH=300
    IMG_HEIGHT=300
    NUM_CLASSES=len(CLASSES)

    data=tf.keras.utils.image_dataset_from_directory(DATA_PATH, image_size=(IMG_WIDTH, IMG_HEIGHT))
    data=data.map(lambda x, y: (x/255, y))

    train_size = int(len(data)*0.8)
    val_size = int(len(data)*0.1)
    test_size = int(len(data)*0.1)

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size+val_size).take(test_size)

    model=tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES)    
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  metrics=['accuracy'])

    with mlflow.start_run():
        history = model.fit(train, epochs=epochs, validation_data=val, batch_size=batch_size)
        
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_metric("accuracy", history.history['accuracy'][-1])
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1])
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.keras.log_model(model, "model")

if __name__ == "__main__":
    typer.run(main)
