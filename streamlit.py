import streamlit as st
from PIL import Image
import tensorflow as tf
import dvc.api
import os

# Load the TensorFlow model
def load_model():
    model_path = dvc.api.get_url('models/my_model.h5')
    model = tf.keras.models.load_model(model_path)
    return model

# Classify the image using the loaded model
def classify_image(image, model):
    img_array = tf.image.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    predicted_class = CLASSES[predictions.argmax()]
    return predicted_class

# Main function
def main():
    st.title("Potato Disease Classifier")

    # Load the model
    model = load_model()

    # Display file uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Classify the image and display the result
        with st.spinner('Classifying...'):
            predicted_class = classify_image(image, model)
            st.success(f'Predicted Class: {predicted_class}')

# Constants
DATA_PATH='data/PotatoPlants'
CLASSES=os.listdir(DATA_PATH)
CLASSES.sort()
IMG_WIDTH=300
IMG_HEIGHT=300
RGB=3
NUM_CLASSES=len(CLASSES)

if __name__ == "__main__":
    main()
