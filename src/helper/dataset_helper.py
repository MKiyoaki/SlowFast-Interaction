import os
import pandas as pd


def get_set_length(file_path):
    data_file = pd.read_csv(file_path)

    return len(data_file)


def get_set_distribution(file_path):
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
    labels = get_set_distribution(file_path)
    length = get_set_length(file_path)
    for key in labels:
        labels[key] /= length
        labels[key] = round(labels[key], 4)

    return labels


# TODO
def check_label_distribution(file_dir):
    os.listdir(file_dir)
    return 0


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

