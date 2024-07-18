import os

year = "2022"

# project dir
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
proj_dir = os.path.join(proj_dir, '../')


# input dir
data_dir = os.path.join(proj_dir, f'../../data/interaction/{year}/')
video_dir = os.path.join(data_dir, 'videos')
label_dir = os.path.join(data_dir, 'labels')


# output dir
log_dir = os.path.join(proj_dir, 'outputs/logs')
model_dir = os.path.join(proj_dir, 'outputs/models')
