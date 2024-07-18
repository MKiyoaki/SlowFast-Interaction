from typing import List, Dict

import pandas as pd
import numpy as np
import torch.utils.data

from pytorchvideo.data import LabeledVideoDataset


class InteractionDataset(LabeledVideoDataset):
    """
    Dataset for video classification with labeled interactions.

    Args:
        labeled_video_paths_file: Path to CSV file with video paths and label paths.
        clip_sampler: Strategy for sampling clips from videos.
        decode_audio: Whether to decode audio from videos.
        decode_video: Whether to decode video frames.
        transform (Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]): Transformations to apply to the clips.
    """

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



