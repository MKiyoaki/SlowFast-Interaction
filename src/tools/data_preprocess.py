import os.path

import torch
import torch.nn.functional as F

from helper.configurations import get_clip_data_dir, get_raw_data_dir
from helper.data_pre_process import dataset_partition, video_clip_by_duration

def split_videos(w_num):
    # Split dataset into train, validation, and test sets
    _, raw_video_dir, raw_label_dir = get_raw_data_dir("2022", w_num)


def split_datasets(w_num):
    data_dir, video_dir, label_dir = get_clip_data_dir("2022", w_num)

