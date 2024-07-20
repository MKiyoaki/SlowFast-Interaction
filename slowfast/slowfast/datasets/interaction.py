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
    video_container as container,
)
from .build import DATASET_REGISTRY
from .random_erasing import RandomErasing
from .transform import create_random_augment, MaskingGenerator, MaskingGenerator3D

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
        assert mode in [
            "train",
            "val",
            "test"
        ], f"Split '{mode}' not supported for InteractionDataset"
        self.mode = mode
        self.cfg = cfg
        self.p_convert_gray = self.cfg.DATA.COLOR_RND_GRAYSCALE
        self.p_convert_dt = self.cfg.DATA.TIME_DIFF_PROB
        self._video_meta = {}
        self._num_retries = num_retries
        self._num_epoch = 0.0
        self._num_yielded = 0
        self.skip_rows = self.cfg.DATA.SKIP_ROWS
        self.use_chunk_loading = (
            True
            if self.mode in ["train"] and self.cfg.DATA.LOADER_CHUNK_SIZE > 0
            else False
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
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )  # train.csv / test.csv / val.csv
        assert pathmgr.exists(path_to_file), "{} dir not found".format(path_to_file)

        self._path_to_videos = []
        self._path_to_labels = []
        self._labels = []
        self._spatial_temporal_idx = []
        self.cur_iter = 0
        self.chunk_epoch = 0
        self.epoch = 0.0
        self.skip_rows = self.cfg.DATA.SKIP_ROWS

        # Use pandas to read the CSV file
        data = pd.read_csv(path_to_file, skiprows=self.skip_rows)

        for clip_idx, row in data.iterrows():
            video_path = row.iloc[0]
            label_path = row.iloc[1]

            for idx in range(self._num_clips):
                self._path_to_videos.append(os.path.join(self.cfg.DATA.PATH_PREFIX, video_path))
                self._path_to_labels.append(os.path.join(self.cfg.DATA.PATH_PREFIX, label_path))
                self._labels.append(self._get_label(self._path_to_labels[-1]))
                self._spatial_temporal_idx.append(idx)
                self._video_meta[clip_idx * self._num_clips + idx] = {}

        assert (
                len(self._path_to_videos) > 0
        ), "Failed to load Interaction split {} from {}".format(
            self.mode, path_to_file
        )
        logger.info(
            "Constructing interaction dataloader (size: {} skip_rows {}) from {} ".format(
                len(self._path_to_videos), self.skip_rows, path_to_file
            )
        )

    def __len__(self):
        return len(self._path_to_videos)

    # TODO: FIX THIS METHOD
    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatedly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (list): the labels of the current video within the sampled clip's time range.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """

        short_cycle_idx = None
        if isinstance(index, tuple):
            index, self._num_yielded = index
            if self.cfg.MULTIGRID.SHORT_CYCLE:
                index, short_cycle_idx = index

        if self.dummy_output is not None:
            return self.dummy_output

        if self.mode in ["train", "val"]:
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                min_scale = int(
                    round(float(min_scale) * crop_size / self.cfg.MULTIGRID.DEFAULT_S)
                )
        elif self.mode in ["test"]:
            temporal_sample_index = (
                    self._spatial_temporal_idx[index] // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            spatial_sample_index = (
                (self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS)
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                     + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))

        num_decode = (
            self.cfg.DATA.TRAIN_CROP_NUM_TEMPORAL if self.mode in ["train"] else 1
        )
        min_scale, max_scale, crop_size = [min_scale], [max_scale], [crop_size]
        if len(min_scale) < num_decode:
            min_scale += [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * (
                    num_decode - len(min_scale)
            )
            max_scale += [self.cfg.DATA.TRAIN_JITTER_SCALES[1]] * (
                    num_decode - len(max_scale)
            )
            crop_size += (
                [self.cfg.MULTIGRID.DEFAULT_S] * (num_decode - len(crop_size))
                if self.cfg.MULTIGRID.LONG_CYCLE or self.cfg.MULTIGRID.SHORT_CYCLE
                else [self.cfg.DATA.TRAIN_CROP_SIZE] * (num_decode - len(crop_size))
            )
            assert self.mode in ["train", "val"]

        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )
                if self.mode not in ["test"]:
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            if video_container is None:
                logger.warning(
                    "Failed to meta load video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 8:
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            frames_decoded, time_idx_decoded = (
                [None] * num_decode,
                [None] * num_decode,
            )

            num_frames = [self.cfg.DATA.NUM_FRAMES]
            sampling_rate = utils.get_random_sampling_rate(
                self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
                self.cfg.DATA.SAMPLING_RATE,
            )
            sampling_rate = [sampling_rate]
            if len(num_frames) < num_decode:
                num_frames.extend(
                    [num_frames[-1] for i in range(num_decode - len(num_frames))]
                )
                sampling_rate.extend(
                    [sampling_rate[-1] for i in range(num_decode - len(sampling_rate))]
                )
            elif len(num_frames) > num_decode:
                num_frames = num_frames[:num_decode]
                sampling_rate = sampling_rate[:num_decode]

            if self.mode in ["train"]:
                assert len(min_scale) == len(max_scale) == len(crop_size) == num_decode

            target_fps = self.cfg.DATA.TARGET_FPS
            if self.cfg.DATA.TRAIN_JITTER_FPS > 0.0 and self.mode in ["train"]:
                target_fps += random.uniform(0.0, self.cfg.DATA.TRAIN_JITTER_FPS)

            frames, time_idx, tdiff = decoder.decode(
                video_container,
                sampling_rate,
                num_frames,
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=(
                    self._video_meta[index] if len(self._video_meta) < 5e6 else {}
                ),
                target_fps=target_fps,
                backend=self.cfg.DATA.DECODING_BACKEND,
                use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
                max_spatial_scale=(
                    min_scale[0] if all(x == min_scale[0] for x in min_scale) else 0
                ),
                time_diff_prob=self.p_convert_dt if self.mode in ["train"] else 0.0,
                temporally_rnd_clips=True,
                min_delta=self.cfg.CONTRASTIVE.DELTA_CLIPS_MIN,
                max_delta=self.cfg.CONTRASTIVE.DELTA_CLIPS_MAX,
            )
            frames_decoded = frames
            time_idx_decoded = time_idx

            if frames_decoded is None or None in frames_decoded:
                logger.warning(
                    "Failed to decode video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if (
                    self.mode not in ["test"]
                    and (i_try % (self._num_retries // 8)) == 0
                ):
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            num_aug = (
                self.cfg.DATA.TRAIN_CROP_NUM_SPATIAL * self.cfg.AUG.NUM_SAMPLE
                if self.mode in ["train"]
                else 1
            )
            num_out = num_aug * num_decode
            f_out, time_idx_out = [None] * num_out, [None] * num_out
            idx = -1
            # TODO: Not quite sure if this is correct
            label = self._labels[index]
            # labels = self._get_frame_labels(self._get_frame_timestamp(time_idx[0][0]), label)

            for i in range(num_decode):
                for _ in range(num_aug):
                    idx += 1
                    f_out[idx] = frames_decoded[i].clone()
                    time_idx_out[idx] = time_idx_decoded[i, :]

                    f_out[idx] = f_out[idx].float()
                    f_out[idx] = f_out[idx] / 255.0

                    if self.mode in ["train"] and self.cfg.DATA.SSL_COLOR_JITTER:
                        f_out[idx] = transform.color_jitter_video_ssl(
                            f_out[idx],
                            bri_con_sat=self.cfg.DATA.SSL_COLOR_BRI_CON_SAT,
                            hue=self.cfg.DATA.SSL_COLOR_HUE,
                            p_convert_gray=self.p_convert_gray,
                            moco_v2_aug=self.cfg.DATA.SSL_MOCOV2_AUG,
                            gaussan_sigma_min=self.cfg.DATA.SSL_BLUR_SIGMA_MIN,
                            gaussan_sigma_max=self.cfg.DATA.SSL_BLUR_SIGMA_MAX,
                        )

                    if self.aug and self.cfg.AUG.AA_TYPE:
                        aug_transform = create_random_augment(
                            input_size=(f_out[idx].size(1), f_out[idx].size(2)),
                            auto_augment=self.cfg.AUG.AA_TYPE,
                            interpolation=self.cfg.AUG.INTERPOLATION,
                        )

                        # T H W C -> T C H W.
                        f_out[idx] = f_out[idx].permute(0, 3, 1, 2)
                        list_img = self._frame_to_list_img(f_out[idx])
                        list_img = aug_transform(list_img)
                        f_out[idx] = self._list_img_to_frames(list_img)
                        f_out[idx] = f_out[idx].permute(0, 2, 3, 1)

                    # Perform color normalization.
                    f_out[idx] = utils.tensor_normalize(
                        f_out[idx], self.cfg.DATA.MEAN, self.cfg.DATA.STD
                    )

                    f_out[idx] = f_out[idx].permute(3, 0, 1, 2)

                    scl, asp = (
                        self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
                        self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
                    )
                    relative_scales = (
                        None if (self.mode not in ["train"] or len(scl) == 0) else scl
                    )
                    relative_aspect = (
                        None if (self.mode not in ["train"] or len(asp) == 0) else asp
                    )
                    f_out[idx] = utils.spatial_sampling(
                        f_out[idx],
                        spatial_idx=spatial_sample_index,
                        min_scale=min_scale[i],
                        max_scale=max_scale[i],
                        crop_size=crop_size[i],
                        random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                        inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                        aspect_ratio=relative_aspect,
                        scale=relative_scales,
                        motion_shift=(
                            self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
                            if self.mode in ["train"]
                            else False
                        ),
                    )

                    if self.rand_erase:
                        erase_transform = RandomErasing(
                            self.cfg.AUG.RE_PROB,
                            mode=self.cfg.AUG.RE_MODE,
                            max_count=self.cfg.AUG.RE_COUNT,
                            num_splits=self.cfg.AUG.RE_COUNT,
                            device="cpu",
                        )
                        f_out[idx] = erase_transform(
                            f_out[idx].permute(1, 0, 2, 3)
                        ).permute(1, 0, 2, 3)

                    f_out[idx] = utils.pack_pathway_output(self.cfg, f_out[idx])
                    if self.cfg.AUG.GEN_MASK_LOADER:
                        mask = self._gen_mask()
                        f_out[idx] = f_out[idx] + [torch.Tensor(), mask]

            frames = f_out[0] if num_out == 1 else f_out
            time_idx = np.array(time_idx_out)

            if (
                num_aug * num_decode > 1
                and not self.cfg.MODEL.MODEL_NAME == "ContrastiveModel"
            ):
                label = [label] * num_aug * num_decode
                index = [index] * num_aug * num_decode
            if self.cfg.DATA.DUMMY_LOAD:
                if self.dummy_output is None:
                    self.dummy_output = (frames, label, index, time_idx, {})

            return frames, label, index, time_idx, {}
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries; trial {}".format(
                    self._num_retries, index
                )
            )

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
                    'RobotInterruption': row['RobotInterruption'],
                    'RobotNonResponding': row['RobotNonResponding'],
                    'RobotInappropriateResponse': row['RobotInappropriateResponse'],
                }

        label_tensor = torch.tensor([label.get(key, 0) for key in label.keys()], dtype=torch.float32)
        # Create a binary string based on feature values
        binary_string = ''.join(str(int(label.get(key, 0))) for key in label.keys())

        # Convert the binary string to a decimal integer
        label_int = int(binary_string, 2)
        return label_int

    def _get_chunk(self, file_obj, chunk_size):
        """
        Helper function to get a chunk of data from a file object.
        """
        rows = []
        for _ in range(chunk_size):
            row = file_obj.readline()
            if not row:
                break
            rows.append(row.strip())
        return rows

    def _set_epoch_num(self, epoch):
        """
        Set the epoch number.
        Args:
            epoch (int): the current epoch number.
        """
        self.epoch = epoch

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
