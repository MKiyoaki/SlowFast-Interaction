#!/usr/bin/env python3

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torchvision import transforms

from slowfast.utils.env import pathmgr
from slowfast.utils.logging import get_logger
from .build import DATASET_REGISTRY
from .random_erasing import RandomErasing
from .transform import create_random_augment

logger = get_logger(__name__)


@DATASET_REGISTRY.register()
class Interaction(torch.utils.data.Dataset):
    """
    Interaction dataset loader. Construct the Interaction dataset loader, then sample
    clips from the videos.
    """

    def __init__(self, cfg, mode, num_retries=100):
        """
        Construct the Interaction video loader with a given csv file. The format of
        the csv file is:
        ```
        video_path label_path
        ...
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options include `train`, `val`, or `test` mode.
            num_retries (int): number of retries.
        """
        assert mode in ["train", "val", "test"], f"Split '{mode}' not supported for InteractionDataset"
        self.mode = mode
        self.cfg = cfg
        self.num_retries = num_retries
        self._video_meta = {}
        self._num_epoch = 0.0
        self._num_yielded = 0
        self.skip_rows = self.cfg.DATA.SKIP_ROWS
        self.use_chunk_loading = (
            True if self.mode in ["train"] and self.cfg.DATA.LOADER_CHUNK_SIZE > 0 else False
        )
        self.dummy_output = None
        self.aug = False
        self.rand_erase = False

        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS

        logger.info(f"Constructing InteractionDataset {mode}...")
        self._construct_loader()

        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, f"{self.mode}.csv")
        assert pathmgr.exists(path_to_file), f"{path_to_file} dir not found"

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []

        with pathmgr.open(path_to_file, "r") as f:
            rows = f.read().splitlines()
            for line in rows:
                video_path, label_path = line.split(self.cfg.DATA.PATH_LABEL_SEPARATOR)
                self._path_to_videos.append(os.path.join(self.cfg.DATA.PATH_PREFIX, video_path))
                self._labels.append(label_path)
                for idx in range(self._num_clips):
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[len(self._path_to_videos) * self._num_clips + idx] = {}

        assert len(self._path_to_videos) > 0, f"Failed to load InteractionDataset split {self.mode} from {path_to_file}"
        logger.info(
            f"Constructing InteractionDataset dataloader (size: {len(self._path_to_videos)}, skip_rows {self.skip_rows}) from {path_to_file}")

    def _set_epoch_num(self, epoch):
        self._num_epoch = epoch

    # def __getitem__(self, index):
    #     """
    #     Given the video index, return the list of frames, label, and video
    #     index if the video can be fetched and decoded successfully, otherwise
    #     repeatedly find a random video that can be decoded as a replacement.
    #     Args:
    #         index (int): the video index provided by the pytorch sampler.
    #     Returns:
    #         frames (tensor): the frames of sampled from the video.
    #         label (int): the label of the current video.
    #         index (int): if the video provided by pytorch sampler can be
    #             decoded, then return the index of the video.
    #     """
    #     if self.dummy_output is not None:
    #         return self.dummy_output


    def _frame_to_list_img(self, tensor_frames):
        """
        Convert the tensor frames to list of images.
        """
        list_img = [tensor_frames[i].permute(1, 2, 0).numpy() for i in range(tensor_frames.size(0))]
        return list_img

    def _list_img_to_frames(self, list_img):
        """
        Convert list of images to tensor frames.
        """
        return torch.stack([torch.tensor(img).permute(2, 0, 1) for img in list_img])

    def __len__(self):
        return len(self._path_to_videos)

    def __repr__(self):
        return f"InteractionDataset(mode={self.mode}, num_clips={self._num_clips})"
