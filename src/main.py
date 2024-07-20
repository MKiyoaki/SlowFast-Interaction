import os.path

import torch
import torch.nn.functional as F


from helper.configurations import raw_data_dir, raw_video_dir, raw_label_dir, video_dir, label_dir, data_dir, proj_dir
from helper.interaction_data_process import dataset_split, video_clip_by_duration


def main():
    # Split dataset into train, validation, and test sets
    # video_clip_by_duration(raw_video_dir, raw_label_dir, data_dir)
    train_file, val_file, test_file = dataset_split(video_dir, label_dir, data_dir)

    return 0


if __name__ == '__main__':
    main()
