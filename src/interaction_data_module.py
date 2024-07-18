import os.path

import pytorchvideo.data
import torch
import torch.utils.data

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)

from helper.configurations import data_dir
from helper.interaction_dataloader import InteractionDataset


class InteractionDataModule:
    # Dataset configuration
    _NUM_WORKERS = 2

    def __init__(self, data_dir, batch_size, clip_duration, train_file, test_file, val_file):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.clip_duration = clip_duration

        self.train_file = train_file
        self.test_file = test_file
        self.val_file = val_file

    def train_dataloader(self):
        """
        TODO: ADD DOCUMENTATION HERE
        """

        train_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(8),
                            Lambda(lambda x: x / 255.0),
                            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                            RandomShortSideScale(min_size=256, max_size=320),
                            RandomCrop(244),
                            RandomHorizontalFlip(p=0.5)
                        ]
                    ),
                ),
            ]
        )

        train_dataset = InteractionDataset(
            self.train_file,
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", self.clip_duration),
            decode_audio=False,
            decode_video=True,
            transform=train_transform,
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self._NUM_WORKERS,
        ), train_dataset

    def val_dataloader(self):
        val_dataset = InteractionDataset(
            self.val_file,
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", self.clip_duration),
            decode_audio=False,
            decode_video=True
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self._NUM_WORKERS,
        )

    def test_dataloader(self):
        test_dataset = InteractionDataset(
            self.test_file,
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", self.clip_duration),
            decode_audio=False,
            decode_video=True
        )
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self._NUM_WORKERS,
        )
