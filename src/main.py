import os.path

import torch
import torch.nn.functional as F

import slowfast.models as models

from helper.configurations import video_dir, label_dir, data_dir
from helper.interaction_data_process import dataset_split
from interaction_classification_module import InteractionClassificationModule
from interaction_data_module import InteractionDataModule


def main():
    # Split dataset into train, validation, and test sets
    train_file, val_file, test_file = dataset_split(video_dir, label_dir, data_dir)

    # Initialize the model
    model = models.ResNet
    criterion = F.cross_entropy

    classy_module = InteractionClassificationModule(model=model)
    data_module = InteractionDataModule(
        data_dir=data_dir,
        batch_size=32,
        clip_duration=2,
        train_file=train_file,
        val_file=val_file,
        test_file=test_file
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    return 0


if __name__ == '__main__':
    main()
