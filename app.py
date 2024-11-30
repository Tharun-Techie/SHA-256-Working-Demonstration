import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Load Pre-trained ResNet50 Model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Function to extract features
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
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
            feature = extract_features(img_path, model)
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
    st.write("Upload an image to find similar trading patterns.")

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

    # Upload query image
    query_image_file = st.file_uploader("Upload Query Image", type=["jpg", "png", "jpeg"])

    if query_image_file is not None:
        # Display query image
        st.image(query_image_file, caption="Query Image", use_column_width=True)

        # Save the uploaded image temporarily
        query_image_path = "query_image.jpg"
        with open(query_image_path, "wb") as f:
            f.write(query_image_file.getbuffer())

        # Perform similarity search if dataset is loaded
        if dataset_features is not None:
            st.write("Searching for similar images...")
            similar_images = find_similar_images(query_image_path, dataset_features, dataset_paths, model)

            # Display results
            st.write(f"Top {len(similar_images)} Similar Images:")
            for img_path, similarity in similar_images:
                st.image(img_path, caption=f"Similarity: {similarity:.2f}", use_column_width=True)
        else:
            st.error("Please provide a valid dataset path in the sidebar.")

if __name__ == "__main__":
    main()
