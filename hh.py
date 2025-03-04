import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
MODEL_PATH = "model.h5" 
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
CLASS_NAMES = ['Healthy', 'Early Blight', 'Late Blight']  # Update with actual labels

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((204, 224))  # Resize to match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Streamlit UI
st.title("🥔 Potato Leaf Disease Classification")
st.write("Upload an image of a potato leaf to predict its condition.")

# File uploader
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    img_array = preprocess_image(img)
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    # Display the result
    st.success(f'Prediction: **{predicted_class}**')
    st.info(f'Confidence: **{confidence:.2f}%**')
