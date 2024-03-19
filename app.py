import streamlit as st
import tensorflow as tf
import numpy as np

# Set the title of the web app
st.title('Skin Type Classification')

# Load the saved model
model = tf.keras.models.load_model('oily_dry.h5')

# Create a file uploader for the test image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Perform the prediction when an image is uploaded
if uploaded_file is not None:
    img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x /= 255.0  # Normalize pixel values to [0, 1]
    x = np.expand_dims(x, axis=0)
    
    # Classify the image
    predictions = model.predict(x)

    # Get the predicted class label
    predicted_class = np.argmax(predictions)

    # Display the prediction results
    st.write("Prediction Results:")
    st.write(f"You have {predictions[0][0]*100:.2f}% Dry Skin")
    st.write(f"You have {predictions[0][1]*100:.2f}% Normal Skin")
    st.write(f"You have {predictions[0][2]*100:.2f}% Oily Skin")
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
