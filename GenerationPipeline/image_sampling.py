import os
import random
import torch
import clip
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans


def sample_from_folder(folder_path, min_images=1, max_images=5):
    """
    Randomly samples between min_images and max_images from the specified folder.
    :param folder_path: Path to the folder containing images.
    :return: List of tuples containing PIL Image objects and their filenames.
    """
    all_images = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    sampled_image_paths = random.sample(
        all_images, random.randint(min_images, max_images)
    )
    return [
        (Image.open(img_path), os.path.basename(img_path))
        for img_path in sampled_image_paths
    ]


# Function to extract CLIP embeddings for images
def get_clip_embeddings(image_paths, model, preprocess):
    embeddings = []
    valid_paths = []

    for img_path in image_paths:
        try:
            image = preprocess(Image.open(img_path)).unsqueeze(0)
            with torch.no_grad():
                embedding = model.encode_image(image).cpu().numpy().flatten()
            embeddings.append(embedding)
            valid_paths.append(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    return np.array(embeddings), valid_paths


def sample_from_cluster(folder_path, num_clusters):
    clustered_csv_path = "clustered_images.csv"

    if os.path.exists(clustered_csv_path):
        print("clustered_images.csv already exists. Skipping clustering.")
        clustered_images = pd.read_csv(clustered_csv_path)
    else:
        print("clustered_images.csv not found. Proceeding with clustering...")
        clip_model, preprocess = clip.load("ViT-B/32")

        # Load data
        image_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]

        # Get image embeddings
        print("Extracting image embeddings using CLIP...")
        embeddings, valid_image_paths = get_clip_embeddings(
            image_paths, clip_model, preprocess
        )

        print(f"Clustering images into {num_clusters} clusters...")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        clustered_images = pd.DataFrame(
            {"image_path": valid_image_paths, "cluster": cluster_labels}
        )
        clustered_images.to_csv(clustered_csv_path, index=False)

    num_samples_per_cluster = random.randint(
        1, 2
    )  # Adjust the number of images to sample from each cluster
    sampled_images = []

    for cluster_id in range(num_clusters):
        cluster_images = clustered_images[clustered_images["cluster"] == cluster_id][
            "image_path"
        ].tolist()
        sampled_image_paths = random.sample(
            cluster_images, min(num_samples_per_cluster, len(cluster_images))
        )
        sampled_images.extend(
            [
                (Image.open(img_path), os.path.basename(img_path))
                for img_path in sampled_image_paths
            ]
        )

    return sampled_images
