import streamlit as st
import tensorflow as tf
import numpy as np
import PIL

# Load the saved model
model = tf.keras.models.load_model('oily_dry.h5')

# Define the function for image classification
def classify_image(image_path):
    # Load the image to be classified
    image = PIL.Image.open(image_path).resize((224, 224))

    # Preprocess the image
    image = tf.keras.preprocessing.image.img_to_array(image)  # Convert to NumPy array
    image /= 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Classify the image
    predictions = model.predict(image)

    # Get the predicted class label
    predicted_class = np.argmax(predictions)

    # Return the predicted class label and percentages
    return predicted_class, float(predictions[0][0]), float(predictions[0][1])

# Define the Streamlit app
def app():
    st.title("Image Classification App")
    st.write("This app classifies an image as either dry or oily skin.")

    # Get the image file from the user
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Classify the image and display the result
    if image_file is not None:
        image_path = image_file.name
        with open(image_path, "wb") as f:
            f.write(image_file.getbuffer())

        st.image(image_path, width=250)

        predicted_class, dry_percentage, oily_percentage = classify_image(image_path)

        dry_percentage = dry_percentage # Replace with your actual dry percentage value
        oily_percentage = oily_percentage  # Replace with your actual oily percentage value

        # Ensure percentages are within the valid range (0 to 100)
        dry_percentage = max(0, min(dry_percentage, 1))
        oily_percentage = max(0, min(oily_percentage, 1))

        # Convert percentages to strings with two decimal places
        dry_percentage_str = f"{dry_percentage:.2f}"
        oily_percentage_str = f"{oily_percentage:.2f}"

        st.progress(dry_percentage,text="You have "+dry_percentage_str+"% Dry Skin")

        st.progress(oily_percentage,text="You have "+oily_percentage_str+"% Oily Skin")

# Run the Streamlit app
if __name__ == "__main__":
    app()