from typing import List, Dict

import pandas as pd
import numpy as np
import torch.utils.data

from pytorchvideo.data import LabeledVideoDataset

"""
TODO: ADD THE DOCUMENTATION
"""


class InteractionDataset(LabeledVideoDataset):
    def __init__(self, labeled_video_paths_file, clip_sampler, decode_audio, decode_video, transform=None):
        super().__init__(
            labeled_video_paths_file,
            clip_sampler,
            decode_audio=decode_audio,
            decode_video=decode_video,
            transform=transform
        )

        labeled_video_paths_df = pd.read_csv(labeled_video_paths_file)
        self._labeled_videos = labeled_video_paths_df['video_path'].tolist()
        label_files = labeled_video_paths_df['label_path'].tolist()

        # Load labels
        self._labels_by_video = self.load_all_labels(self._labeled_videos, label_files)

    def __iter__(self):
        self._video_sampler_iter = None
        worker_info = torch.utils.data.get_worker_info()
        if self._video_random_generator is not None and worker_info is not None:
            base_seed = worker_info.seed - worker_info.id
            self._video_random_generator.manual_seed(base_seed)
        return self

    @staticmethod
    def load_all_labels(video_paths: List[str], label_files: List[str]) -> Dict[str, List[Dict]]:
        labels_by_video = {}
        for video_path, label_file in zip(video_paths, label_files):
            # Ensure the video path is a key in the dictionary
            video_name = video_path[-13:-4]
            if video_name not in labels_by_video:
                labels_by_video[video_name] = []
            labels = InteractionDataset.load_labels(label_file)
            labels_by_video[video_name].extend(labels)
        return labels_by_video

    @staticmethod
    def load_labels(file_path: str) -> List[Dict]:
        data = pd.read_csv(file_path)

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



