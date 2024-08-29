import cv2
import os
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34
from torchvision.transforms import transforms
from collections import defaultdict


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, self.image_paths[idx]


def extract_features(model, dataloader, device):
    features = []
    image_paths = []
    model.eval()
    with torch.no_grad():
        for imgs, paths in dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            features.append(outputs.cpu().numpy())
            image_paths.extend(paths)
    return np.concatenate(features), image_paths


def kmeans_clustering_images(input_dir, output_dir, k):
    """
    Execute K-means clustering for the images in the input directory. Save the results to the output directory.

    Args:
        input_dir (str): Path to the input directory containing images.
        output_dir (str): Path to the output directory where clusters will be saved.
        k (int): Number of clusters.

    Returns:
        None
    """

    # Make the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Normalization for the images
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load a pretrained ResNet model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet34(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])  # Remove the last FC layer
    model = model.to(device)

    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if
                   f.endswith(".jpg") or f.endswith(".png")]

    if not image_paths:
        raise ValueError("No valid images found in the input directory.")

    dataset = ImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    features, image_paths = extract_features(model, dataloader, device)
    features = features.reshape(features.shape[0], -1)  # Reshape as a 2D vector

    # Execute k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(features)

    clusters = defaultdict(list)
    for label, image_path in zip(labels, image_paths):
        clusters[label].append(image_path)

    # Save the images for each clusters
    for label, image_list in clusters.items():
        cluster_dir = os.path.join(output_dir, f"cluster_{label}")
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)
        for image_path in image_list:
            image_name = os.path.basename(image_path)
            cv2.imwrite(os.path.join(cluster_dir, image_name), cv2.imread(image_path))

    # Visualized the results
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    plt.figure(figsize=(10, 8))
    for i in range(k):
        cluster_points = reduced_features[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}")

    plt.legend()
    plt.title("UserAwkwardness Clustering Results")
    plt.show()