import os
import pandas as pd

from helper.configurations import get_clip_data_dir


def get_set_length(file_path):
    data_file = pd.read_csv(file_path)

    return len(data_file)


def get_all_set_distribution(yyyy, ww):
    res = {
        'UserAwkwardness': 0,
        'RobotMistake': 0,
        'RobotInterruption': 0,
        'RobotNonResponding': 0,
        'RobotInappropriateResponse': 0,
        'InteractionRupture': 0
    }

    _, _, label_dir = get_clip_data_dir(yyyy, ww)

    labels = os.listdir(label_dir)

    # Function to count labels
    def count_labels(label_files):
        for label_file in label_files:
            df = pd.read_csv(os.path.join(label_dir, label_file))
            for label in res.keys():
                res[label] += df[label].sum()

    # Count labels in all sets
    count_labels(labels)

    return res


def get_set_distribution(file_path):
    """
    Get the distribution of all the classes of labels for a dataset file.
    A dataset is a .csv file contains two columns: path to data, path to label.
    The results will be presented as the exact number stored in a dictionary.
    Args:
        file_path: The path to the annotation file.
    Return:
        labels: The dictionary stored with the number of all classes in the annotation file.
    """
    data_file = pd.read_csv(file_path)

    label_counts = {}

    # Iterate through each row in the dataset file
    for idx, row in data_file.iterrows():
        label_path = row['label_path']

        # Check if the label file exists
        if os.path.exists(label_path):
            label_file = pd.read_csv(label_path)

            # Iterate through each label in the label file
            for col in label_file.columns:
                if col not in ['start_time', 'end_time']:  # Ignore non-label columns
                    for label in label_file[col]:
                        if col not in label_counts:
                            label_counts[col] = 0
                        label_counts[col] += label
        else:
            print(f"[Warning] Label file {label_path} does not exist.")

    return label_counts


def get_set_distribution_percentage(file_path):
    """
    Get the distribution of all the classes of labels for a dataset file.
    A dataset is a .csv file contains two columns: path to data, path to label.
    The results will be presented as the percentage stored in a dictionary.
    Args:
        file_path: The path to the annotation file.
    Return:
        labels: The dictionary stored with the percentage results of all classes in the annotation file.
    """
    labels = get_set_distribution(file_path)
    length = get_set_length(file_path)
    for key in labels:
        labels[key] /= length
        labels[key] = round(labels[key], 4)

    return labels


def generate_combined_labels(file_paths, output_path):
    combined_df = pd.DataFrame()

    for file_path in file_paths:
        try:
            # Read the label file
            df = pd.read_csv(file_path)
            # Append the data to the combined DataFrame
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    try:
        # Write the combined DataFrame to the output path
        combined_df.to_csv(output_path, index=False)
        print(f"Combined labels file saved to {output_path}")
    except Exception as e:
        print(f"Error writing to {output_path}: {e}")



def get_set_no_overlapping(file_path, targets):
    label_counts = {target: 0 for target in targets}

    data_file = pd.read_csv(file_path)

    # Iterate through each row in the dataset file
    for idx, row in data_file.iterrows():
        label_path = row['label_path']

        # Check if the label file exists
        if os.path.exists(label_path):
            label_file = pd.read_csv(label_path)

            # Iterate through each row in the label file
            for _, label_row in label_file.iterrows():
                valid_row = True
                found_target = None

                for target in targets:
                    label_value = label_row.get(target, 0)

                    if label_value == 1:
                        if found_target is None:
                            found_target = target
                        else:
                            valid_row = False
                            break
                    elif label_value != 0:
                        valid_row = False
                        break

                # If the row is valid and the found target is set
                if valid_row and found_target:
                    label_counts[found_target] += 1
        else:
            print(f"[Warning] Label file {label_path} does not exist.")

    return label_counts
