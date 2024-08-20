import os

from helper.configurations import get_raw_data_dir, get_clip_data_dir, get_frame_data_dir
from helper.dataset_helper import (
    get_set_distribution,
    get_set_distribution_percentage,
    get_set_no_overlapping,
)


def check(yyyy="2022"):
    for i in range(1, 5):
        ww = f"w{i}"
        data_dir, video_dir, label_dir = get_frame_data_dir(yyyy, ww)

        print(get_set_distribution(os.path.join(data_dir, "vis.csv")))
        print(get_set_distribution_percentage(os.path.join(data_dir, "vis.csv")))
        # print(get_set_no_overlapping(os.path.join(data_dir, "vis.csv"), ["UserAwkwardness", "RobotMistake"]))


if __name__ == "__main__":
    check(yyyy="2022")
