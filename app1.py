import os
import io
import pyperclip
from PIL import Image, ImageGrab
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# Load Pre-trained ResNet50 Model
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Function to extract features
def extract_features(img, model):
    img = img.resize((224, 224))
    img_data = np.expand_dims(np.array(img), axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()

# Process dataset to extract features
def process_dataset(image_folder, model):
    features = []
    image_paths = []
    for img_file in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_file)
        try:
            img = Image.open(img_path).convert("RGB")
            feature = extract_features(img, model)
            features.append(feature)
            image_paths.append(img_path)
        except:
            print(f"Error processing image: {img_path}")
    return np.array(features), image_paths

# Find similar images
def find_similar_images(query_image, dataset_features, dataset_paths, model, top_n=5):
    query_features = extract_features(query_image, model)
    similarities = cosine_similarity([query_features], dataset_features)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [(dataset_paths[i], similarities[i]) for i in top_indices]

# Main Streamlit app
def main():
    st.title("Trading Chart Pattern Finder")
    st.write("Paste an image from the clipboard or upload a file to find similar trading patterns.")

    # Sidebar to configure dataset path
    dataset_path = st.sidebar.text_input("Dataset Folder Path", "path_to_images")
    if not os.path.exists(dataset_path):
        st.sidebar.error("The provided dataset folder does not exist.")

    # Extract features from dataset if a valid folder is provided
    dataset_features, dataset_paths = None, None
    if os.path.exists(dataset_path):
        st.sidebar.write("Processing dataset...")
        dataset_features, dataset_paths = process_dataset(dataset_path, model)
        st.sidebar.success("Dataset processed!")

    # Clipboard or file upload options
    query_image = None
    st.subheader("Input Image")

    if st.button("Paste from Clipboard"):
        try:
            clipboard_img = ImageGrab.grabclipboard()  # Grab image from clipboard
            if clipboard_img is not None:
                query_image = clipboard_img.convert("RGB")
                st.image(query_image, caption="Pasted from Clipboard", use_column_width=True)
            else:
                st.warning("No image found in the clipboard.")
        except Exception as e:
            st.error(f"Error accessing clipboard: {e}")

    # Upload query image as a fallback
    query_image_file = st.file_uploader("Or Upload Query Image", type=["jpg", "png", "jpeg"])
    if query_image_file is not None:
        query_image = Image.open(query_image_file).convert("RGB")
        st.image(query_image, caption="Uploaded Image", use_column_width=True)

    # Perform similarity search if dataset is loaded
    if query_image and dataset_features is not None:
        st.write("Searching for similar images...")
        similar_images = find_similar_images(query_image, dataset_features, dataset_paths, model)

        # Display results
        st.write(f"Top {len(similar_images)} Similar Images:")
        for img_path, similarity in similar_images:
            st.image(img_path, caption=f"Similarity: {similarity:.2f}", use_column_width=True)
    elif query_image:
        st.warning("Please provide a valid dataset path in the sidebar.")

if __name__ == "__main__":
    main()
