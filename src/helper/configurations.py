import os

year = "2022"
w_num = "w1"

# project dir
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
proj_dir = os.path.join(proj_dir, '../')


# input dir
raw_data_dir = os.path.join(proj_dir, f'data/interaction/raw/{year}/{w_num}')
raw_video_dir = os.path.join(raw_data_dir, 'videos')
raw_label_dir = os.path.join(raw_data_dir, 'labels')

data_dir = os.path.join(proj_dir, f'data/interaction/clip/{year}/{w_num}')
video_dir = os.path.join(data_dir, 'videos')
label_dir = os.path.join(data_dir, 'labels')


# output dir
log_dir = os.path.join(proj_dir, 'outputs/logs')
model_dir = os.path.join(proj_dir, 'outputs/models')
