import streamlit as st
import tensorflow as tf
import numpy as np
import PIL

model = tf.keras.models.load_model('oily_dry.h5')

def classify_image(image_path):
    # Load the image to be classified
    image = PIL.Image.open(image_path).resize((224, 224))

    # Preprocess the image
    image = tf.keras.preprocessing.image.img_to_array(image)  # Convert to NumPy array
    image /= 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Classify the image
    predictions = model.predict(image)

    # Get the predicted class label and percentages
    predicted_class, dry_percentage, oily_percentage, normal_percentage = np.argmax(predictions), float(predictions[0][0]), float(predictions[0][1]), float(predictions[0][2])

    # Ensure percentages are within the valid range (0 to 100)
    dry_percentage = max(0, min(dry_percentage, 1))
    oily_percentage = max(0, min(oily_percentage, 1))
    normal_percentage = max(0, min(normal_percentage, 1))

    # Convert percentages to strings with two decimal places
    dry_percentage_str = f"{dry_percentage*100:.2f}"
    oily_percentage_str = f"{oily_percentage*100:.2f}"
    normal_percentage_str = f"{normal_percentage*100:.2f}"

    # Return the predicted class label and percentages
    return predicted_class, dry_percentage_str, oily_percentage_str, normal_percentage_str

def app():
    st.title("Oily/Dry Skin Level Predictor 🧑🏻👩🏻")
    st.write("Coded by Manith Jayaba")

    st.write("This app can measure the oiliness and dryness of your skin")

    # Get the image file from the user
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Classify the image and display the result
    if image_file is not None:
        image_path = image_file.name
        with open(image_path, "wb") as f:
            f.write(image_file.getbuffer())

        predicted_class, dry_percentage, oily_percentage, normal_percentage = classify_image(image_path)

        st.image(image_path, width=250)

        st.progress(dry_percentage,text="You have "+dry_percentage+"% Dry Skin")
        st.progress(oily_percentage,text="You have "+oily_percentage+"% Oily Skin")
        st.progress(normal_percentage,text="You have "+normal_percentage+"% Normal Skin")