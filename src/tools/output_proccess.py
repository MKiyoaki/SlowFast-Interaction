import os
from collections import defaultdict

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from helper.configurations import output_dir
from helper.video_helper import combine_videos, combine_frames, extract_frames_from_video, convert_video_to_frame


def combine_output(yyyy, label, model_name):
    """
    Combine the results of the two pathway outputs from Grad-CAM. This method will combine the videos from slow path
    into a video sequence, and combine the images from fast path into a large grid image.
    """

    if label == "ua":
        label_full = "UserAwkwardness"
    elif label == "rm":
        label_full = "RobotMistake"
    else:
        return 0

    combine_videos(f"{output_dir}/interaction/{yyyy}/slowfast/{label}/{model_name}/grad_cam/Path_1/{label_full}",
                  f"{output_dir}/interaction/{yyyy}/slowfast/{label}/{model_name}/grad_cam/")

    combine_frames(f"{output_dir}/interaction/{yyyy}/slowfast/{label}/{model_name}/grad_cam/Path_0/{label_full}",
                   f"{output_dir}/interaction/{yyyy}/slowfast/{label}/{model_name}/grad_cam/")


def kmeans_clustering_images(input_dir, output_dir, k):
    """
    Execute K-means clustering for the images of the input directory. Given the results to the output directory.

    Args:
        input_dir (string):
        output_dir (string):
        k (int): K value for the clustering.
    Return:
        None
    """

    # Create the output dir if it is not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def extract_features(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))  # 调整图片大小
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        return hist.flatten()

    features = []
    image_paths = []
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # 处理图片文件
            image = cv2.imread(file_path)
            if image is not None:
                features.append(extract_features(image))
                image_paths.append(file_path)
        elif filename.endswith(".mp4") or filename.endswith(".avi") or filename.endswith(".mov"):
            # 处理视频文件
            frames = extract_frames_from_video(file_path, 1)
            for i, frame in enumerate(frames):
                features.append(extract_features(frame))
                image_paths.append(f"{file_path}_frame_{i}.png")

        # 检查是否有提取到特征
    if len(features) == 0:
        raise ValueError("No valid images or frames found in the input directory.")

    features = np.array(features)
    if features.ndim == 1:
        raise ValueError("Expected 2D array for features, but got 1D array instead.")

    # Execute K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(features)

    clusters = defaultdict(list)
    for label, image_path in zip(labels, image_paths):
        clusters[label].append(image_path)

    for label, image_list in clusters.items():
        cluster_dir = os.path.join(output_dir, f"cluster_{label}")
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)
        for image_path in image_list:
            image_name = os.path.basename(image_path)
            cv2.imwrite(os.path.join(cluster_dir, image_name), cv2.imread(image_path))

    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    plt.figure(figsize=(10, 8))
    for i in range(k):
        cluster_points = reduced_features[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}")

    plt.legend()
    plt.title("Image Clusters")
    plt.show()


if __name__ == "__main__":
    yyyy = "2022"
    ww = "w1"
    label = "ua"
    model_name = "res_w4_step"

    label_full = ""
    if label == "ua":
        label_full = "UserAwkwardness"
    elif label == "rm":
        label_full = "RobotMistake"

    input = f"{output_dir}/interaction/{yyyy}/slowfast/{label}/{model_name}/grad_cam/Path_0/{label_full}/../imgs"
    output = f"{output_dir}/interaction/{yyyy}/slowfast/{label}/{model_name}/grad_cam/"

    # combine_output(yyyy, label, model_name)
    #  convert_video_to_frame(input)
    kmeans_clustering_images(input, output, 4)
