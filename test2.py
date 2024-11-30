from sklearn.metrics.pairwise import cosine_similarity

# Function to find similar images
def find_similar_images(query_image_path, dataset_features, dataset_paths, model, top_n=5):
    query_features = extract_features(query_image_path, model)
    similarities = cosine_similarity([query_features], dataset_features)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [(dataset_paths[i], similarities[i]) for i in top_indices]

query_image = 'path_to_query_image.jpg'
similar_images = find_similar_images(query_image, dataset_features, dataset_paths, model)

# Display results
for img_path, similarity in similar_images:
    print(f"Image: {img_path}, Similarity: {similarity}")
