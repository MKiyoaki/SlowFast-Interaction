import os
import csv
import shutil

import pandas as pd
import time

from imblearn.combine import SMOTETomek
from moviepy.video.io.VideoFileClip import VideoFileClip
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MultiLabelBinarizer


def video_clip_by_duration(raw_data_path, raw_label_path, clip_data_path):
    """
    Split the raw video into clips according to the duration annotation in the label files. The outputs will
    be stored in the clip_data_path folder.

    Args:
        raw_data_path (string): Path to the raw videos.
        raw_label_path (string): Path to the raw label files.
        clip_data_path (string): Path to the clip data.
    """
    # Create output directory if not exists
    video_output_path = os.path.join(clip_data_path, 'videos')
    label_output_path = os.path.join(clip_data_path, 'labels')
    os.makedirs(video_output_path, exist_ok=True)
    os.makedirs(label_output_path, exist_ok=True)

    label_files = os.listdir(raw_label_path)
    for label_file in label_files:
        # Load labels
        file_name = label_file.split("/")[-1].split(".")[0]
        labels = get_label(os.path.join(raw_label_path, label_file))

        # Process each label and create corresponding video and label files
        for idx, label in enumerate(labels):
            start_time = label['start_time']
            end_time = label['end_time']
            output_video_path = os.path.join(video_output_path,
                                             f"{file_name}_{idx + 1}.avi")
            output_label_path = os.path.join(label_output_path,
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


def video_clip_by_time_step(raw_data_path, raw_label_path, clip_data_path, time_threshold=1, time_step=1):
    """
    Further split the video from duration into clips according to the time step and threshold value. The outputs
    will be stored in the clip_data_path folder.
    The time step means the final length for each clip. Threshold means the least length for videos to get splitting,
    otherwise the video will be ignored.

    Args:
        raw_data_path (string): Path to the raw videos.
        raw_label_path (string): Path to the raw label files.
        clip_data_path (string): Path to the clip data.
        time_threshold (int): Threshold for the time step.
        time_step (int): The time step for splitting, or the final length for each clip.
     """
    # Create output directory if not exists
    video_output_path = os.path.join(clip_data_path, 'videos')
    label_output_path = os.path.join(clip_data_path, 'labels')
    os.makedirs(video_output_path, exist_ok=True)
    os.makedirs(label_output_path, exist_ok=True)

    for video_file in os.listdir(raw_data_path):
        if video_file.endswith(('.avi', '.mp4', '.mov', '.mkv')):  # Check if the files are videos
            video_path = os.path.join(raw_data_path, video_file)
            label_file = os.path.splitext(video_file)[0] + '.csv'
            label_path = os.path.join(raw_label_path, label_file)

            # Open files
            with VideoFileClip(video_path) as video:
                video_duration = video.duration  # Get the video length

                if video_duration > time_threshold:
                    num_clips = int(video_duration // time_step)

                    for i in range(num_clips):
                        start_time = i * time_step
                        end_time = start_time + time_step

                        if end_time > video_duration:
                            break

                        clip = video.subclip(start_time, end_time)

                        new_video_name = f"{os.path.splitext(video_file)[0]}_{i + 1}.avi"
                        new_video_path = os.path.join(video_output_path, new_video_name)
                        clip.write_videofile(new_video_path, codec="libx264", audio_codec="aac")

                        # Copy the corresponding label files
                        if os.path.exists(label_path):
                            new_label_name = f"{os.path.splitext(video_file)[0]}_{i + 1}.csv"
                            new_label_path = os.path.join(label_output_path, new_label_name)
                            shutil.copyfile(label_path, new_label_path)


def video_clip(raw_data_path, raw_label_path, clip_data_path, time_threshold=1, time_step=1):
    """
    Split the raw video into clips directly based on the label duration information, and further clip within
    those durations by a fixed time step. If the current clip is within a label's duration, the corresponding
    label will be attached to the clip.
    This is the combination of previous two methods.

    Args:
        raw_data_path (string): Path to the raw videos.
        raw_label_path (string): Path to the raw label files.
        clip_data_path (string): Path to the output clips.
        time_threshold (float): Minimum duration threshold for further splitting.
        time_step (float): Time step for splitting clips further.
    """
    # Create output directory if not exists
    video_output_path = os.path.join(clip_data_path, 'videos')
    label_output_path = os.path.join(clip_data_path, 'labels')
    os.makedirs(video_output_path, exist_ok=True)
    os.makedirs(label_output_path, exist_ok=True)

    label_files = os.listdir(raw_label_path)
    for label_file in label_files:
        # Load labels
        file_name = os.path.splitext(label_file)[0]
        labels = get_label(os.path.join(raw_label_path, label_file))

        with VideoFileClip(os.path.join(raw_data_path, f"{file_name}.avi")) as video:
            for idx, label in enumerate(labels):
                start_time = label['start_time']
                end_time = label['end_time']
                duration = end_time - start_time

                if duration > time_threshold:
                    # Further split based on time_step if duration exceeds time_threshold
                    num_clips = int(duration // time_step)

                    for i in range(num_clips):
                        segment_start = start_time + i * time_step
                        segment_end = segment_start + time_step
                        if segment_end > end_time:
                            segment_end = end_time

                        clip = video.subclip(segment_start, segment_end)

                        new_video_name = f"{file_name}_{idx + 1}_part_{i + 1}.avi"
                        new_video_path = os.path.join(video_output_path, new_video_name)
                        clip.write_videofile(new_video_path, codec="libx264", audio_codec="aac")

                        # Save the label corresponding to the clip
                        segment_label = pd.DataFrame([label['features']])
                        segment_label['start_time'] = segment_start
                        segment_label['end_time'] = segment_end
                        segment_label.to_csv(os.path.join(label_output_path, f"{file_name}_{idx + 1}_{i + 1}.csv"),
                                             index=False)
                else:
                    # If the duration is less than or equal to the threshold, no further splitting
                    clip = video.subclip(start_time, end_time)
                    new_video_name = f"{file_name}_{idx + 1}.avi"
                    new_video_path = os.path.join(video_output_path, new_video_name)
                    clip.write_videofile(new_video_path, codec="libx264", audio_codec="aac")

                    # Save the label corresponding to the clip
                    label_data = pd.DataFrame([label['features']])
                    label_data['start_time'] = start_time
                    label_data['end_time'] = end_time
                    label_data.to_csv(os.path.join(label_output_path, f"{file_name}_{idx + 1}.csv"), index=False)

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


def dataset_get_vis(output_path, data_path, label_path):
    """
    Create a dataset called vis.csv, contains all the videos without any oversampling or undersampling results.
    This is used for generating Grad-CAM results.

    Args：
        output_path (str): the path to the output directory.
        data_path (str): the path to the data directory.
        label_path (str): the path to the label file.
    """
    # Read video and label names
    video_names = os.listdir(data_path)
    label_names = os.listdir(label_path)

    # Create a mapping of video to labels
    video_to_label = []
    for video_name in video_names:
        label_file = find_label_file(video_name, label_names)
        if label_file:
            video_path = os.path.join(data_path, video_name)
            label_path_full = os.path.join(label_path, label_file)
            video_to_label.append((video_path, label_path_full))
        else:
            print(f"[{time.time()}][Error] Label file not found for {video_name}.")

    # Write the complete list to a CSV file
    output_csv_path = os.path.join(output_path, "vis.csv")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video_path', 'label_path'])
        writer.writerows(video_to_label)

    return output_csv_path


def undersample(df):
    # Separate positive and negative samples
    # Assuming target_labels is a list of labels we are interested in
    # We assume the labels column contains a tuple of (label_name, label_value)
    pos_samples = df[df['labels'].apply(lambda x: any(val == 1 for label, val in x))]
    neg_samples = df[~df['labels'].apply(lambda x: any(val == 1 for label, val in x))]

    # Determine the number of samples to keep
    min_samples = min(len(pos_samples), len(neg_samples))

    # Undersample both positive and negative samples
    pos_samples = pos_samples.sample(min_samples, random_state=42)
    neg_samples = neg_samples.sample(min_samples, random_state=42)

    # Combine the undersampled positive and negative samples
    undersampled_df = pd.concat([pos_samples, neg_samples])

    return undersampled_df


def oversample(df):
    # Separate positive and negative samples
    pos_samples = df[df['labels'].apply(lambda x: any(val == 1 for label, val in x))]
    neg_samples = df[~df['labels'].apply(lambda x: any(val == 1 for label, val in x))]

    # Determine the number of samples to match the larger class
    max_samples = max(len(pos_samples), len(neg_samples))

    # Oversample the smaller class
    pos_samples = pos_samples.sample(max_samples, replace=True, random_state=42)
    neg_samples = neg_samples.sample(max_samples, replace=True, random_state=42)

    # Combine the oversampled positive and negative samples
    oversampled_df = pd.concat([pos_samples, neg_samples])

    return oversampled_df



def smote_tomek_sampling(df):
    # TODO: Doesn't work correctly.
    # Flatten the label dictionaries for compatibility with SMOTE/ENN
    df_flat = pd.DataFrame(df['labels'].tolist(), index=df.index)

    # Convert label columns to a binary (one-hot) format
    mlb = MultiLabelBinarizer()
    df_flat_encoded = pd.DataFrame(mlb.fit_transform(df_flat.apply(lambda x: tuple(x.items()), axis=1)),
                                   columns=mlb.classes_,
                                   index=df.index)

    sampler = SMOTETomek(random_state=42)

    # Perform the SMOTE-Tomek sampling
    X_resampled, y_resampled = sampler.fit_resample(df_flat_encoded, df.index)

    # Mapping back to original indices
    resampled_df = df.iloc[df.index.get_indexer(y_resampled)].copy()

    # Update the 'labels' column to match resampled data
    resampled_df['labels'] = X_resampled.apply(lambda x: dict(zip(mlb.classes_, x)), axis=1)

    return resampled_df


def dataset_partition(
            data_path,
            label_path,
            output_path,
            target_labels,
            sampling="oversample",
            train_scales=0.7,
            val_scales=0.15,
            test_scales=0.15
    ):
    """
    Dataset partition methods. During this process user can choose a sampling strategy. This will produce three files
    that contains the path of data in train, test, val sets.

    Args:
        data_path (str): the path to the data folder.
        label_path (str): the path to the label file.
        output_path (str): the path to the output folder.
        target_labels (list): the target labels.
        sampling (str): the sampling strategy. Can only be chosen from "none", "mixed", "oversample" or "undersample".
        train_scales (float): the train scale factor.
        val_scales (float): the validation scale factor.
        test_scales (float): the test scale factor.
    Returns:
        train_csv_path (str): the path to the train csv file.
        val_csv_path (str): the path to the val csv file.
        test_csv_path (str): the path to the test csv file.
    """
    video_names = os.listdir(data_path)
    label_names = os.listdir(label_path)

    assert train_scales + val_scales + test_scales == 1

    video_to_label = {}
    for video_name in video_names:
        label_file = find_label_file(video_name, label_names)
        if label_file:
            labels_df = pd.read_csv(os.path.join(label_path, label_file))
            # Extract only the label columns (ignore start_time and end_time)
            label_columns = target_labels
            video_to_label[video_name] = labels_df[label_columns].iloc[0].to_dict()

    # Create a DataFrame from video_to_label
    df = pd.DataFrame(
        list(video_to_label.items()),
        columns=['video_name', 'labels']
    )
    df['labels'] = df['labels'].apply(lambda x: tuple(x.items()))

    # Split the data into training and temporary (validation + test)
    train_df, temp_df = train_test_split(df, test_size=(1 - train_scales), stratify=df['labels'])
    val_df, test_df = train_test_split(temp_df, test_size=(test_scales / (val_scales + test_scales)), stratify=temp_df['labels'])

    # Perform oversampling to balance the dataset
    if sampling == "mixed":
        train_df = smote_tomek_sampling(train_df)
    elif sampling == "oversample":
        train_df = oversample(train_df)
    elif sampling == "undersample":
        train_df = undersample(train_df)
    elif sampling == "none":
        pass
    else:
        print(f"Error! No sampling named as: {sampling}. ")
        return 0

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
    """
    Helper method for dataset partition. This could get the name of label file according to the name of
    video file.

    Args:
        video_name (str): the name of the video.
        label_names (list): the label names.
    Return:
        label_name (str): the name of the label.
    """
    video_basename = os.path.splitext(video_name)[0]  # remove extension
    for label_name in label_names:
        if label_name.startswith(video_basename):
            return label_name
    return None
