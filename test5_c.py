import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from io import BytesIO
from PIL import ImageGrab


# Step 1: Initialize ResNet50 for feature extraction
def load_model():
    base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model


# Step 2: Preprocess the image for ResNet50
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


# Step 3: Extract features from an image
def extract_features(image, model):
    preprocessed = preprocess_image(image)
    features = model.predict(preprocessed)
    return features.flatten()


# Step 4: Load dataset and extract features
def load_dataset(dataset_path, model):
    features = []
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(("png", "jpg", "jpeg")):
                image_path = os.path.join(root, file)
                image = Image.open(image_path).convert("RGB")
                feature = extract_features(image, model)
                features.append(feature)
                image_paths.append(image_path)
    return np.array(features), image_paths


# Step 5: Find the most similar images using cosine similarity
def find_similar_images(query_features, dataset_features, image_paths, top_n=5):
    similarities = cosine_similarity([query_features], dataset_features)[0]
    indices = np.argsort(similarities)[::-1][:top_n]
    similar_images = [(image_paths[i], similarities[i]) for i in indices]
    return similar_images


# Step 6: Streamlit Web Interface
def main():
    st.title("Trading Chart Pattern Matching")
    st.write("Upload an image or paste a screenshot to find similar patterns.")

    # Sidebar for dataset selection
    dataset_path = st.sidebar.text_input("Dataset Folder", value="dataset/")
    model = load_model()

    if "dataset_features" not in st.session_state:
        if os.path.exists(dataset_path):
            st.write("Loading dataset and extracting features...")
            dataset_features, image_paths = load_dataset(dataset_path, model)
            st.session_state["dataset_features"] = dataset_features
            st.session_state["image_paths"] = image_paths
            st.write("Dataset loaded successfully!")
        else:
            st.write("Please provide a valid dataset folder path.")

    # Image input (via clipboard or upload)
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    clipboard_image = st.button("Paste from Clipboard")

    query_image = None
    if clipboard_image:
        clipboard_data = ImageGrab.grabclipboard()
        if clipboard_data is not None:
            query_image = clipboard_data
        else:
            st.write("No image found in the clipboard.")

    if uploaded_image:
        query_image = Image.open(uploaded_image)

    if query_image:
        st.image(query_image, caption="Query Image", use_container_width=True)

        if "dataset_features" in st.session_state:
            query_features = extract_features(query_image.convert("RGB"), model)
            similar_images = find_similar_images(
                query_features,
                st.session_state["dataset_features"],
                st.session_state["image_paths"],
            )

            # Display similar images
            st.write("Most similar images:")
            for path, similarity in similar_images:
                st.image(path, caption=f"Similarity: {similarity:.4f}", use_container_width=True)
        else:
            st.write("Dataset features are not loaded. Please check the dataset path.")


if __name__ == "__main__":
    main()
