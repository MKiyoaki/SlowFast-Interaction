import os

year = "2022"
w_num = "w4"

# project dir
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
proj_dir = os.path.join(proj_dir, '../')


# input dir
def get_raw_data_dir(year, w_num):
    raw_data_dir = os.path.join(proj_dir, f'data/interaction/raw/{year}/{w_num}')
    raw_video_dir = os.path.join(raw_data_dir, 'videos')
    raw_label_dir = os.path.join(raw_data_dir, 'labels')

    return raw_data_dir, raw_video_dir, raw_label_dir

def get_clip_data_dir(year, w_num):
    data_dir = os.path.join(proj_dir, f'data/interaction/clip/{year}/{w_num}')
    video_dir = os.path.join(data_dir, 'videos')
    label_dir = os.path.join(data_dir, 'labels')

    return data_dir, video_dir, label_dir


# output dir
log_dir = os.path.join(proj_dir, 'outputs/logs')
model_dir = os.path.join(proj_dir, 'outputs/models')
