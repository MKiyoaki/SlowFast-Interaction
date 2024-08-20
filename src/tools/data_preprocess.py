import os.path

from helper.configurations import get_clip_data_dir, get_raw_data_dir, get_frame_data_dir
from helper.data_pre_process import video_clip_by_duration, video_clip_by_time_step


def split_videos():
    """
    Split the raw video data into frame clips.

    TODO: Integrate the two methods into one.
    """

    for i in range(1, 5):
        w_num = f"w{i}"

        _, raw_video_dir, raw_label_dir = get_raw_data_dir("2022", w_num)
        clip_data_dir, video_dir, label_dir = get_clip_data_dir("2022", w_num)
        frame_data_dir, _, _ = get_frame_data_dir("2022", w_num)

        # video_clip_by_duration(raw_video_dir, raw_label_dir, clip_data_dir)
        video_clip_by_time_step(video_dir, label_dir, frame_data_dir, time_threshold=1, time_step=1)


if __name__ == '__main__':
    split_videos()
