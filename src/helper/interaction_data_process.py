import os
import random
import csv
import pandas as pd
import time

from moviepy.video.io.VideoFileClip import VideoFileClip
from sklearn.model_selection import train_test_split


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


def dataset_split(data_path, label_path, output_path, train_scales=0.8, val_scales=0.1, test_scales=0.1):
    """
    Create annotated csv files containing video paths and their labels. Split by
    0.8 : 0.1 : 0.1 for train, validation, and test sets with stratified sampling.
    :param data_path: Path to video data directory
    :param label_path: Path to label data directory
    :param output_path: Path to save CSV collection files
    :param train_scales: Scale of training dataset
    :param val_scales: Scale of validation dataset
    :param test_scales: Scale of testing dataset
    :return: Paths to train, val, test CSV files
    """

    # Step 1: Read video and label names
    video_names = os.listdir(data_path)
    label_names = os.listdir(label_path)

    # Step 2: Create a mapping of video to labels
    video_to_label = {}
    for video_name in video_names:
        label_file = find_label_file(video_name, label_names)
        if label_file:
            labels_df = pd.read_csv(os.path.join(label_path, label_file))
            # Extract only the label columns (ignore start_time and end_time)
            label_columns = ['UserAwkwardness', 'RobotMistake', 'RobotNonResponding']
            video_to_label[video_name] = labels_df[label_columns].iloc[0].to_dict()  # Adjust if necessary

    # Create a DataFrame from video_to_label
    df = pd.DataFrame(list(video_to_label.items()), columns=['video_name', 'labels'])

    # Step 3: Convert labels to a categorical format for stratification
    df['labels'] = df['labels'].apply(lambda x: tuple(x.items()))  # Convert label dict to tuple for stratification

    # Split the data into training and temporary (validation + test)
    train_df, temp_df = train_test_split(df, test_size=(1 - train_scales), stratify=df['labels'])

    # Further split the temporary set into validation and test
    val_df, test_df = train_test_split(temp_df, test_size=(test_scales / (val_scales + test_scales)), stratify=temp_df['labels'])

    # Step 4: Write filenames to corresponding CSV files with labels
    def write_to_csv(file_path, df):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['video_path', 'label_path'])
            for _, row in df.iterrows():
                video_path = os.path.join(data_path, row['video_name'])
                label_file = find_label_file(row['video_name'], label_names)
                if label_file is None:
                    print(f"[{time.time()}][Error] Label file not found for {row['video_name']}.")
                    continue
                writer.writerow([video_path, os.path.join(label_path, label_file)])
        return file_path

    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    train_csv_path = os.path.join(output_path, "train.csv")
    val_csv_path = os.path.join(output_path, "val.csv")
    test_csv_path = os.path.join(output_path, "test.csv")

    write_to_csv(train_csv_path, train_df)
    write_to_csv(val_csv_path, val_df)
    write_to_csv(test_csv_path, test_df)

    return train_csv_path, val_csv_path, test_csv_path

def find_label_file(video_name, label_names):
    video_basename = os.path.splitext(video_name)[0]  # remove extension
    for label_name in label_names:
        if label_name.startswith(video_basename):
            return label_name
    return None