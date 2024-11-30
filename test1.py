import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

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

# Extract features for all images in the dataset
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

dataset_features, dataset_paths = process_dataset('path_to_images', model)
