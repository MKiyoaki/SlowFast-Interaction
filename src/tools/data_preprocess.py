import os.path

import torch
import torch.nn.functional as F


from helper.configurations import raw_data_dir, raw_video_dir, raw_label_dir, video_dir, label_dir, data_dir
from helper.interaction_data_process import dataset_split, video_clip_by_duration

def split_videos(w_num):
    # Split dataset into train, validation, and test sets
    video_clip_by_duration(raw_video_dir, raw_label_dir, data_dir)

def split_datasets(w_num):

    dataset_split(video_dir, label_dir, data_dir)
