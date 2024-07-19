import os.path

import torch
import torch.nn.functional as F

import slowfast.models as models

from detectron2.config import get_cfg

from helper.configurations import video_dir, label_dir, data_dir, proj_dir
from helper.interaction_data_process import dataset_split
from interaction_classification_module import InteractionClassificationModule
from interaction_data_module import InteractionDataModule


def main():
    # Split dataset into train, validation, and test sets
    train_file, val_file, test_file = dataset_split(video_dir, label_dir, data_dir)


    # Initialize the model
    criterion = F.cross_entropy
    loss_fn = torch.nn.CrossEntropyLoss()

    data_module = InteractionDataModule(
        data_dir=data_dir,
        batch_size=32,
        clip_duration=1,
        train_file=train_file,
        val_file=val_file,
        test_file=test_file
    )

    return 0


if __name__ == '__main__':
    main()
