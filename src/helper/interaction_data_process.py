import os
import random
import csv
import time

from pyslowfast.slowfast.utils.logging import setup_logging


def dataset_split(data_path, label_path, collection_path, train_scales=0.8, val_scales=0.1, test_scales=0.1):
    """
    Create annotated csv files containing video paths and their labels

    :param data_path: Path to video data directory
    :param label_path: Path to label data directory
    :param collection_path: Path to save CSV collection files
    :param train_scales: Scale of training dataset
    :param val_scales: Scale of validation dataset
    :param test_scales: Scale of testing dataset
    :return: Paths to train, val, test CSV files
    """

    # Step 1: Read video and label names
    video_names = os.listdir(data_path)
    label_names = os.listdir(label_path)

    # Step 2: Check if collection_path exists, if not, create it
    if not os.path.exists(collection_path):
        os.makedirs(collection_path)

    # Step 3: Shuffle indices for random splitting
    current_data_length = len(video_names)
    current_data_index_list = list(range(current_data_length))
    random.shuffle(current_data_index_list)

    # Step 4: Calculate split sizes
    train_size = int(train_scales * current_data_length)
    val_size = int(val_scales * current_data_length)
    test_size = current_data_length - train_size - val_size

    # Step 5: Split indices into train, val, test
    train_indices = current_data_index_list[:train_size]
    val_indices = current_data_index_list[train_size:train_size + val_size]
    test_indices = current_data_index_list[train_size + val_size:]

    # Function to find label file corresponding to video file
    def find_label_file(video_name):
        video_basename = os.path.splitext(video_name)[0]  # remove extension
        for label_name in label_names:
            if label_name.startswith(video_basename):
                return label_name
        return None

    # Step 6: Write filenames to corresponding CSV files with labels
    def write_to_csv(file_path, files):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['video_path', 'label_path'])
            for video_name in files:
                label_file = find_label_file(video_name)
                if label_file is None:
                    print(f"[{time.time}][Error] Label file not found for {video_name}. Operation aborted. ")
                    return None
                video_path = os.path.join(data_path, video_name)
                writer.writerow([video_path, os.path.join(label_path, label_file)])
        return file_path


    train_files = [video_names[idx] for idx in train_indices]
    val_files = [video_names[idx] for idx in val_indices]
    test_files = [video_names[idx] for idx in test_indices]

    train_csv_path = os.path.join(collection_path, "train.csv")
    val_csv_path = os.path.join(collection_path, "val.csv")
    test_csv_path = os.path.join(collection_path, "test.csv")

    train_csv = write_to_csv(train_csv_path, train_files)
    val_csv = write_to_csv(val_csv_path, val_files)
    test_csv = write_to_csv(test_csv_path, test_files)

    return train_csv, val_csv, test_csv
