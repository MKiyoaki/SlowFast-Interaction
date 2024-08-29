import json
import os
import shutil

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from helper.configurations import output_dir
from helper.kmeans_helper import kmeans_clustering_images
from helper.video_helper import combine_videos, combine_frames, convert_video_to_frame


def combine_output(yyyy, label, model_name):
    """
    Combine the results of the two pathway outputs from Grad-CAM. This method will combine the videos from slow path
    into a video sequence, and combine the images from fast path into a large grid image.

    Args:
        yyyy: year of the datasets.
        label: label of the interaction ruptures.
        model_name: model name.
    """

    if label == "ua":
        label_full = "UserAwkwardness"
    elif label == "rm":
        label_full = "RobotMistake"
    else:
        return 0

    combine_videos(f"{output_dir}/interaction/{yyyy}/slowfast/{label}/{model_name}/grad_cam/Path_1/{label_full}_True",
                  f"{output_dir}/interaction/{yyyy}/slowfast/{label}/{model_name}/grad_cam/")

    combine_frames(f"{output_dir}/interaction/{yyyy}/slowfast/{label}/{model_name}/grad_cam/Path_0/{label_full}_True",
                   f"{output_dir}/interaction/{yyyy}/slowfast/{label}/{model_name}/grad_cam/")


if __name__ == "__main__":
    yyyy = "2022"
    ww = "w1"
    label = "ua"
    model_name = "res_w1_step"

    label_full = ""
    if label == "ua":
        label_full = "UserAwkwardness"
    elif label == "rm":
        label_full = "RobotMistake"

    input = f"{output_dir}/interaction/{yyyy}/slowfast/{label}/{model_name}/grad_cam2/Path_0/{label_full}_True/"
    output = f"{output_dir}/interaction/{yyyy}/slowfast/{label}/{model_name}/grad_cam2/clustering"

    combine_output(yyyy, label, model_name)
    # convert_video_to_frame(input)
    # kmeans_clustering_images(os.path.join(input, "../imgs"), output, 3)
