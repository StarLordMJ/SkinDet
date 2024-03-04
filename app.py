import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model (ensure correct path for loading)
model = tf.keras.models.load_model('oily_dry.h5')

def classify_image(image_path):
    # Load and resize the image
    image = Image.open(image_path).resize((224, 224))

    # Preprocess the image
    image = tf.keras.preprocessing.image.img_to_array(image) / 255.0  # Combine steps
    image = np.expand_dims(image, axis=0)

    # Classify the image
    predictions = model.predict(image)

    # Process predictions for clarity
    predicted_class = np.argmax(predictions)  # Get class label
    percentages = predictions[0] * 100  # Get percentages directly
    dry_percentage, normal_percentage, oily_percentage = percentages

    # Ensure percentages are within 0-100
    dry_percentage = max(0, min(dry_percentage, 100))
    oily_percentage = max(0, min(oily_percentage, 100))
    normal_percentage = max(0, min(normal_percentage, 100))

    return predicted_class, dry_percentage, oily_percentage, normal_percentage

def app():
    st.title("Oily/Dry Skin Level Predictor 🧑🏻👩🏻")
    
    st.write("Coded by Manith Jayaba")

    st.write("This app can measure the oiliness and dryness of your skin")

    # Get the image file
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Classify and display the result
    if image_file is not None:
        image_path = image_file.name
        with open(image_path, "wb") as f:
            f.write(image_file.getbuffer())

        predicted_class, dry_percentage, oily_percentage, normal_percentage = classify_image(image_path)

        st.image(image_path, width=250)

        # Convert percentages to integers before passing to progress
        st.progress(int(dry_percentage), text=f"You have {dry_percentage:.2f}% Dry Skin")
        st.progress(int(oily_percentage), text=f"You have {oily_percentage:.2f}% Oily Skin")
        st.progress(int(normal_percentage), text=f"You have {normal_percentage:.2f}% Normal Skin")


if __name__ == "__main__":
    app()
