import os

year = "2022"
w_num = "w4"

# project dir
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
proj_dir = os.path.join(proj_dir, '../')

# output dir
output_dir = os.path.join(proj_dir, 'outputs')


# input dir
def get_raw_data_dir(year, w_num):
    """
    Get the corresponding raw data, video and label directory for the given year and week number.

    Args:
        year: year information
        w_num: week number
    Return:
        data_dir: The path to the raw data directory
        video_dir: The path to the raw video directory
        label_dir: The path to the raw label directory
    """
    raw_data_dir = os.path.join(proj_dir, f'data/interaction/raw/{year}/{w_num}')
    raw_video_dir = os.path.join(raw_data_dir, 'videos')
    raw_label_dir = os.path.join(raw_data_dir, 'labels')

    return raw_data_dir, raw_video_dir, raw_label_dir


def get_clip_data_dir(year, w_num):
    """
    Get the corresponding clip data, video and label directory for the given year and week number.

    Args:
        year: year information
        w_num: week number
    Return:
        data_dir: The path to the clip data directory
        video_dir: The path to the clip video directory
        label_dir: The path to the clip label directory
    """
    data_dir = os.path.join(proj_dir, f'data/interaction/clip/{year}/{w_num}')
    video_dir = os.path.join(data_dir, 'videos')
    label_dir = os.path.join(data_dir, 'labels')

    return data_dir, video_dir, label_dir


def get_frame_data_dir(year, w_num):
    """
    Get the corresponding frame data, video and label directory for the given year and week number.

    Args:
        year: year information
        w_num: week number
    Return:
        data_dir: The path to the frame data directory
        video_dir: The path to the frame video directory
        label_dir: The path to the frame label directory
    """
    frame_data_dir = os.path.join(proj_dir, f'data/interaction/frame/{year}/{w_num}')
    frame_video_dir = os.path.join(frame_data_dir, 'videos')
    frame_label_dir = os.path.join(frame_data_dir, 'labels')

    return frame_data_dir, frame_video_dir, frame_label_dir
