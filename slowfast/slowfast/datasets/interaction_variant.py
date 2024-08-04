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
from . import (
    decoder as decoder,
    transform as transform,
    utils as utils,
    video_container as container, Interaction,
)
from .build import DATASET_REGISTRY
from .random_erasing import RandomErasing
from .transform import create_random_augment, MaskingGenerator, MaskingGenerator3D

logger = get_logger(__name__)


@DATASET_REGISTRY.register()
class Interaction_2023(Interaction):
    """
    Interaction dataset loader. Construct the Interaction dataset loader, then sample
    clips from the videos.
    """

    def __init__(self, cfg, mode, num_retries=100):
        """
        Construct the Interaction video loader with a given csv file. The format of
        the csv file is:
        ```
        video_path_1 label_path_1
        video_path_2 label_path_2
        ...
        video_path_N label_path_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options include `train`, `val`, or `test` mode.
            num_retries (int): number of retries.
        """
        super().__init__(cfg, mode, num_retries)


    def _get_label(self, label_path):
        """
        Load labels for a given video path.
        Args:
            label_path (str): the path to the label file.
        Returns:
            labels (dict): the labels for the video.
        """
        data = pd.read_csv(label_path)
        label = {}
        for idx, row in data.iterrows():
            label = {
                    'UserAwkwardness': row['UserAwkwardness'],
                    'RobotInterruption': row['RobotInterruption'],
                    'RobotNonResponding': row['RobotNonResponding'],
                    'RobotInappropriateResponse': row['RobotInappropriateResponse'],
                }

        # Convert the binary string to a decimal integer
        label_tensor = torch.tensor(list(label.values()), dtype=torch.float32)
        return label_tensor



@DATASET_REGISTRY.register()
class Interaction_ir_2022(Interaction):
    def __init__(self, cfg, mode, num_retries=100):
        """
        Construct the Interaction video loader with a given csv file. The format of
        the csv file is:
        ```
        video_path_1 label_path_1
        video_path_2 label_path_2
        ...
        video_path_N label_path_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options include `train`, `val`, or `test` mode.
            num_retries (int): number of retries.
        """
        super().__init__(cfg, mode, num_retries)


    def _get_label(self, label_path):
        """
        Load labels for a given video path.
        Args:
            label_path (str): the path to the label file.
        Returns:
            labels (dict): the labels for the video.
        """
        data = pd.read_csv(label_path)
        label = {}
        for idx, row in data.iterrows():
            label = {
                    'UserAwkwardness': row['UserAwkwardness'],
                    'RobotMistake': row['RobotMistake'],
            }

        # Take the union of them
        label_tensor = int(label['UserAwkwardness']) or int(label['RobotMistake'])
        return label_tensor


@DATASET_REGISTRY.register()
class Interaction_ua_2022(Interaction):
    def __init__(self, cfg, mode, num_retries=100):
        """
        Construct the Interaction video loader with a given csv file. The format of
        the csv file is:
        ```
        video_path_1 label_path_1
        video_path_2 label_path_2
        ...
        video_path_N label_path_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options include `train`, `val`, or `test` mode.
            num_retries (int): number of retries.
        """
        super().__init__(cfg, mode, num_retries)

    def _get_label(self, label_path):
        """
        Load labels for a given video path.
        Args:
            label_path (str): the path to the label file.
        Returns:
            labels (dict): the labels for the video.
        """
        data = pd.read_csv(label_path)
        label = {}
        for idx, row in data.iterrows():
            label = {
                    'UserAwkwardness': row['UserAwkwardness'],
                }

        label_tensor = int(label['UserAwkwardness'])
        return label_tensor


@DATASET_REGISTRY.register()
class Interaction_rm_2022(Interaction):
    def __init__(self, cfg, mode, num_retries=100):
        """
        Construct the Interaction video loader with a given csv file. The format of
        the csv file is:
        ```
        video_path_1 label_path_1
        video_path_2 label_path_2
        ...
        video_path_N label_path_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options include `train`, `val`, or `test` mode.
            num_retries (int): number of retries.
        """
        super().__init__(cfg, mode, num_retries)

    def _get_label(self, label_path):
        """
        Load labels for a given video path.
        Args:
            label_path (str): the path to the label file.
        Returns:
            labels (dict): the labels for the video.
        """
        data = pd.read_csv(label_path)
        label = {}
        for idx, row in data.iterrows():
            label = {
                    'RobotMistake': row['RobotMistake'],
                }

        label_tensor = int(label['RobotMistake'])
        return label_tensor


@DATASET_REGISTRY.register()
class Interaction_ri_2022(Interaction):
    def __init__(self, cfg, mode, num_retries=100):
        """
        Construct the Interaction video loader with a given csv file. The format of
        the csv file is:
        ```
        video_path_1 label_path_1
        video_path_2 label_path_2
        ...
        video_path_N label_path_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options include `train`, `val`, or `test` mode.
            num_retries (int): number of retries.
        """
        super().__init__(cfg, mode, num_retries)

    def _get_label(self, label_path):
        """
        Load labels for a given video path.
        Args:
            label_path (str): the path to the label file.
        Returns:
            labels (dict): the labels for the video.
        """
        data = pd.read_csv(label_path)
        label = {}
        for idx, row in data.iterrows():
            label = {
                    'RobotInterruption': row['RobotInterruption'],
            }

        label_tensor = int(label['RobotInterruption'])
        return label_tensor


@DATASET_REGISTRY.register()
class Interaction_rnr_2022(Interaction):
    def __init__(self, cfg, mode, num_retries=100):
        """
        Construct the Interaction video loader with a given csv file. The format of
        the csv file is:
        ```
        video_path_1 label_path_1
        video_path_2 label_path_2
        ...
        video_path_N label_path_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options include `train`, `val`, or `test` mode.
            num_retries (int): number of retries.
        """
        super().__init__(cfg, mode, num_retries)


    def _get_label(self, label_path):
        """
        Load labels for a given video path.
        Args:
            label_path (str): the path to the label file.
        Returns:
            labels (dict): the labels for the video.
        """
        data = pd.read_csv(label_path)
        label = {}
        for idx, row in data.iterrows():
            label = {
                    'RobotNonResponding': row['RobotNonResponding'],
                }

        # Convert the binary string to a decimal integer
        label_tensor = int(label['RobotNonResponding'])
        return label_tensor


@DATASET_REGISTRY.register()
class Interaction_rir_2022(Interaction):
    def __init__(self, cfg, mode, num_retries=100):
        """
        Construct the Interaction video loader with a given csv file. The format of
        the csv file is:
        ```
        video_path_1 label_path_1
        video_path_2 label_path_2
        ...
        video_path_N label_path_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options include `train`, `val`, or `test` mode.
            num_retries (int): number of retries.
        """
        super().__init__(cfg, mode, num_retries)


    def _get_label(self, label_path):
        """
        Load labels for a given video path.
        Args:
            label_path (str): the path to the label file.
        Returns:
            labels (dict): the labels for the video.
        """
        data = pd.read_csv(label_path)
        label = {}
        for idx, row in data.iterrows():
            label = {
                    'RobotInappropriateResponse': row['RobotInappropriateResponse'],
                }

        # Convert the binary string to a decimal integer
        label_tensor = int(label['RobotInappropriateResponse'])
        return label_tensor


@DATASET_REGISTRY.register()
class Interaction_unary_2022(Interaction):
    """
    Interaction dataset loader. Construct the Interaction dataset loader, then sample
    clips from the videos.
    """

    def __init__(self, cfg, mode, num_retries=100):
        """
        Construct the Interaction video loader with a given csv file. The format of
        the csv file is:
        ```
        video_path_1 label_path_1
        video_path_2 label_path_2
        ...
        video_path_N label_path_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options include `train`, `val`, or `test` mode.
            num_retries (int): number of retries.
        """
        super().__init__(cfg, mode, num_retries)

    def _get_label(self, label_path):
        """
        Load labels for a given video path.
        Args:
            label_path (str): the path to the label file.
        Returns:
            labels (dict): the labels for the video.
        """
        data = pd.read_csv(label_path)
        binary_str = ''
        for idx, row in data.iterrows():
            # Extract the label values and convert them to binary strings
            user_awkwardness = str(int(row['UserAwkwardness']))
            robot_interruption = str(int(row['RobotInterruption']))
            robot_non_responding = str(int(row['RobotNonResponding']))
            robot_inappropriate_response = str(int(row['RobotInappropriateResponse']))

            # Concatenate these binary strings into a single binary string
            binary_str = (robot_inappropriate_response +
                          robot_non_responding +
                          robot_interruption +
                          user_awkwardness)

            # Convert the binary string to a decimal integer
        if binary_str:
            return int(binary_str, 2)
        else:
            return 0  # Return 0 if binary_str is empty
