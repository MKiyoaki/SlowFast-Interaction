import os.path

import torch
import torch.nn.functional as F


from helper.configurations import raw_data_dir, raw_video_dir, raw_label_dir, video_dir, label_dir, data_dir, proj_dir
from helper.interaction_data_process import dataset_split, video_clip_by_duration
from interaction_data_module import InteractionDataModule


def main():
    # Split dataset into train, validation, and test sets
    video_clip_by_duration(raw_video_dir, raw_label_dir, data_dir)
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
