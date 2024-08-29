import os.path

from helper.configurations import get_clip_data_dir, get_raw_data_dir, get_frame_data_dir
from helper.preprocess_helper import video_clip_by_duration, video_clip_by_time_step, dataset_partition, \
    dataset_get_vis, video_clip
from helper.statistic_helper import generate_combined_labels


def split_videos():
    """
    Split the raw video data into frame clips.

    """
    for i in range(1, 5):
        w_num = f"w{i}"

        _, raw_video_dir, raw_label_dir = get_raw_data_dir("2022", w_num)
        clip_data_dir, video_dir, label_dir = get_clip_data_dir("2022", w_num)
        frame_data_dir, _, _ = get_frame_data_dir("2022", w_num)

        # video_clip_by_duration(raw_video_dir, raw_label_dir, clip_data_dir)
        # video_clip_by_time_step(video_dir, label_dir, frame_data_dir, time_threshold=1, time_step=1)
        video_clip(video_dir, label_dir, frame_data_dir, time_threshold=1, time_step=1) # need test


def generate_dataset():
        """
        Start the data sampling and partition process after the video splitting is done.
        This method will generate the train/test/val sets for each week and for combined 4 weeks.

        """
        yyyy = "2022"
        ww = "w1"

        data_dir, video_dir, label_dir = get_frame_data_dir(yyyy, ww)

        train_labels = []
        test_labels = []
        val_labels = []

        output_dir = os.path.join(data_dir, "../csv/wall")

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for i in range(1, 5):
            yyyy = "2022"
            ww = f"w{i}"

            data_dir_k, video_dir, label_dir = get_frame_data_dir(yyyy, ww)

            dataset_partition(
                video_dir, label_dir, data_dir_k,
                target_labels=["RobotMistake"],
                sampling="oversample",
                train_scales=0.7,
                test_scales=0.15,
                val_scales=0.15
            )

            print(f"Dataset partition is done for: week {i}. ")

            dataset_get_vis(
                data_dir_k,
                video_dir,
                label_dir,
            )

            train_labels.append(os.path.join(data_dir_k, "train.csv"))
            test_labels.append(os.path.join(data_dir_k, "test.csv"))
            val_labels.append(os.path.join(data_dir_k, "val.csv"))

        # Generate the datasets including data from all of 4 weeks.
        generate_combined_labels(train_labels, os.path.join(output_dir, "train.csv"))
        generate_combined_labels(test_labels, os.path.join(output_dir, "test.csv"))
        generate_combined_labels(val_labels, os.path.join(output_dir, "val.csv"))

        return 0


if __name__ == '__main__':
    split_videos()
    generate_dataset()
