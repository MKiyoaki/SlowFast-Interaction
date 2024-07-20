import os
import random
import csv
import pandas as pd
import time

from moviepy.video.io.VideoFileClip import VideoFileClip


def video_clip_by_duration(raw_data_path, raw_label_path, clip_data_path):
    # Create output directory if not exists
    output_video_dir = os.path.join(clip_data_path, "videos/")
    if not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)

    output_label_dir = os.path.join(clip_data_path, "labels/")
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    label_files = os.listdir(raw_label_path)
    for label_file in label_files:
        # Load labels
        file_name = label_file.split("/")[-1].split(".")[0]
        labels = get_label(os.path.join(raw_label_path, label_file))

        # Process each label and create corresponding video and label files
        for idx, label in enumerate(labels):
            start_time = label['start_time']
            end_time = label['end_time']
            output_video_path = os.path.join(output_video_dir,
                                             f"{file_name}_{idx + 1}.avi")
            output_label_path = os.path.join(output_label_dir,
                                             f"{file_name}_{idx + 1}.csv")

            # Cut video segment
            with VideoFileClip(os.path.join(raw_data_path, f"{file_name}.avi")) as video:
                new_clip = video.subclip(start_time, end_time)
                new_clip.write_videofile(output_video_path, codec="libx264")

            # Save label to CSV
            label_data = pd.DataFrame([label['features']])
            label_data['start_time'] = start_time
            label_data['end_time'] = end_time
            label_data.to_csv(output_label_path, index=False)

    return 0


def get_label(label_path):
    """
    Load labels for a given video path.
    Args:
        label_path (str): the path to the label file.
    Returns:
        labels (list): the labels for the video.
    """
    data = pd.read_csv(label_path)
    labels = []
    for idx, row in data.iterrows():
        label = {
            'start_time': row['Begin Time - ss.msec'],
            'end_time': row['End Time - ss.msec'],
            'features': {
                'UserAwkwardness': row['UserAwkwardness'],
                'RobotMistake': row['RobotMistake'],
                'RobotInterruption': row['RobotInterruption'],
                'RobotNonResponding': row['RobotNonResponding'],
                'RobotInappropriateResponse': row['RobotInappropriateResponse'],
            }
        }
        labels.append(label)
    return labels


def dataset_split(data_path, label_path, collection_path, train_scales=0.8, val_scales=0.1, test_scales=0.1):
    """
    Create annotated csv files containing video paths and their labels. Initially split by
    0.8 : 0.1 : 0.1
    for train, validation and test sets.

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
