import os

import torch
import torch.nn.functional as F

from helper.configurations import get_raw_data_dir, get_clip_data_dir
from helper.interaction_data_process import (
    dataset_partition,
    video_clip_by_duration,
    dataset_partition_undersampling,
)
from helper.dataset_helper import (
    get_set_distribution,
    get_set_length,
    get_set_distribution_percentage,
    generate_combined_labels
)


def combine_labels(yyyy, output_path):
    train_labels = []
    test_labels = []
    val_labels = []
    for i in range(4):
        data_dir_k, _, _ = get_clip_data_dir(yyyy, f"w{i}")

        train_labels.append(os.path.join(data_dir_k, "train.csv"))
        test_labels.append(os.path.join(data_dir_k, "test.csv"))
        val_labels.append(os.path.join(data_dir_k, "val.csv"))

    generate_combined_labels(train_labels, os.path.join(output_path, "train.csv"))
    generate_combined_labels(test_labels, os.path.join(output_path, "test.csv"))
    generate_combined_labels(val_labels, os.path.join(output_path, "val.csv"))


def main():
    # Split dataset into train, validation, and test sets
    yyyy = "2022"
    ww = "w4"

    raw_data_dir, raw_video_dir, raw_label_dir = get_raw_data_dir(yyyy, ww)
    data_dir, video_dir, label_dir = get_clip_data_dir(yyyy, ww)

    # video_clip_by_duration(raw_video_dir, raw_label_dir, data_dir)
    # dataset_partition_undersampling(
    #     video_dir, label_dir, data_dir,
    #     target_labels=["UserAwkwardness"],
    #     train_scales=0.8,
    #     test_scales=0.1,
    #     val_scales=0.1
    # )

    # print(get_set_distribution_percentage(os.path.join(data_dir, "train.csv")))
    # print(get_set_distribution_percentage(os.path.join(data_dir, "val.csv")))

    combine_labels(yyyy, os.path.join(data_dir, "../csv/csv_w1to4"))




    return 0


if __name__ == '__main__':
    main()
