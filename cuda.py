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
import tensorflow as tf

# Check for GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

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

# Step 3: Batch preprocessing for feature extraction
def batch_preprocess_images(image_paths, target_size=(224, 224)):
    images = []
    for path in image_paths:
        image = Image.open(path).convert("RGB").resize(target_size)
        image_array = img_to_array(image)
        images.append(image_array)
    images = np.array(images)
    images = preprocess_input(images)
    return images

# Step 4: Extract features in batches
def extract_features_in_batches(image_paths, model, batch_size=32):
    features = []
    num_images = len(image_paths)
    for start in range(0, num_images, batch_size):
        end = min(start + batch_size, num_images)
        batch_images = batch_preprocess_images(image_paths[start:end])
        batch_features = model.predict(batch_images, verbose=0)
        features.extend(batch_features)
    return np.array(features)

# Step 5: Load dataset and extract features
def load_dataset(dataset_path, model, batch_size=32):
    image_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(dataset_path)
        for file in files
        if file.endswith(("png", "jpg", "jpeg"))
    ]
    features = extract_features_in_batches(image_paths, model, batch_size)
    return features, image_paths

# Step 6: Find the most similar images using cosine similarity
def find_similar_images(query_features, dataset_features, image_paths, top_n=5):
    similarities = cosine_similarity([query_features], dataset_features)[0]
    indices = np.argsort(similarities)[::-1][:top_n]
    similar_images = [(image_paths[i], similarities[i]) for i in indices]
    return similar_images

# Step 7: Streamlit Web Interface
def main():
    st.title("Trading Chart Pattern Matching (GPU-Accelerated)")
    st.write("Upload an image or paste a screenshot to find similar patterns.")

    # Sidebar for dataset selection
    dataset_path = st.sidebar.text_input("Dataset Folder", value="dataset/")
    model = load_model()

    if "dataset_features" not in st.session_state:
        if os.path.exists(dataset_path):
            st.write("Loading dataset and extracting features...")
            dataset_features, image_paths = load_dataset(dataset_path, model, batch_size=32)
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
        st.image(query_image, caption="Query Image", use_column_width=True)

        if "dataset_features" in st.session_state:
            query_features = extract_features_in_batches(
                [query_image.convert("RGB")], model, batch_size=1
            )[0]
            similar_images = find_similar_images(
                query_features,
                st.session_state["dataset_features"],
                st.session_state["image_paths"],
            )

            # Display similar images
            st.write("Most similar images:")
            for path, similarity in similar_images:
                st.image(path, caption=f"Similarity: {similarity:.4f}", use_column_width=True)
        else:
            st.write("Dataset features are not loaded. Please check the dataset path.")

if __name__ == "__main__":
    main()
