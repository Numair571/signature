import streamlit as st
from PIL import Image, ImageOps
from keras.models import load_model
import numpy as np
import os

# Load Keras model
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Streamlit UI
st.title('Signature Verification')

# Upload image for Image Classification
image_file = st.file_uploader('Upload Signature image ', type=['png', 'jpg', 'jpeg'])

def load_img(img):
    img = Image.open(img)
    return img

if image_file is not None:
    file_details = {'name': image_file.name, 'size': image_file.size, 'type': image_file.type}
    st.write(file_details)
    st.image(load_img(image_file), width=255)

    with open(os.path.join('uploads', 'image.jpg'), 'wb') as f:
        f.write(image_file.getbuffer())

    st.success('Image Saved')

    # Preprocess image for Keras model
    image = Image.open('uploads/image.jpg').convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict using Keras model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    st.info('Image Classification:')
    st.write("Class:", class_name[2:])
    st.write("Confidence Score:", confidence_score)
