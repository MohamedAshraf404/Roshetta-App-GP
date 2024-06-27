import streamlit as st
import pickle
from PIL import Image
import numpy as np
import os
import numpy as np
import pandas as pd 
import random
import logging
import matplotlib.pyplot as plt
from PIL import Image

# Deep learning libraries
import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Setting seeds for reproducibility
seed = 232
np.random.seed(seed)
tf.random.set_seed(seed)

# Define the model architecture
def build_model(input_shape):
    inputs = Input(shape=input_shape)

    # First conv block
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    # Second conv block
    x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    # Third conv block
    x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    # Fourth conv block
    x = SeparableConv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.2)(x)

    # Fifth conv block
    x = SeparableConv2D(filters=1024, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(filters=1024, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.2)(x)

    # FC layer
    x = Flatten()(x)
    x = Dense(units=1024, activation='relu')(x) 
    x = Dropout(rate=0.7)(x)
    x = Dense(units=512, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=256, activation='relu')(x)  
    x = Dropout(rate=0.3)(x)

    # Output layer
    output = Dense(units=1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=output)
    return model

# Initialize and compile the model
input_shape = (150, 150, 3)
model = build_model(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the weights
model.load_weights('best_weights.hdf5')


# Function to preprocess the image
def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

st.title("Pneumonia Detection from Chest X-ray")

# Upload image
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type="jpeg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-ray.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    result = np.round(prediction).astype(int)[0][0]
    
    if result == 0:
        st.write("The image is **not infected**.")
    else:
        st.write("The image is **infected**.")






























# import streamlit as st
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# from tensorflow.keras.models import load_model

# # Load the model
# @st.cache(allow_output_mutation=True)
# def load_model():
#     model = tf.keras.models.load_model('best_weights.hdf5')
#     return model

# model = load_model()

# # Function to preprocess the image
# def preprocess_image(image):
#     image = image.resize((150, 150))  # Resize the image
#     image = np.array(image) / 255.0  # Normalize the image
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     return image

# # Streamlit app
# st.title('Pneumonia Detection App')

# uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

# if uploaded_file is not None:
#     # Display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image.', use_column_width=True)

#     # Preprocess the image
#     processed_image = preprocess_image(image)

#     # Make predictions
#     prediction = model.predict(processed_image)
#     result = np.round(prediction).astype(int)[0][0]

#     # Interpret the result
#     if result == 0:
#         st.write("Prediction: The image is not infected.")
#     else:
#         st.write("Prediction: The image is infected.")
