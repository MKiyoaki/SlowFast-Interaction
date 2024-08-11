from helper.video_helper import combine_videos, combine_frames

ww = "w4"
approach = "rm"
filename = "1_pre"
label_name = "RobotMistake"


combine_videos(f"/media/kiyoaki/Ventoy/projects/SlowFast-Interaction/outputs/interaction/2022/{ww}/slowfast/{approach}/{filename}/grad_cam/Path_1/{label_name}",
               f"/media/kiyoaki/Ventoy/projects/SlowFast-Interaction/outputs/interaction/2022/{ww}/slowfast/{approach}/{filename}/grad_cam/")

combine_frames(f"/media/kiyoaki/Ventoy/projects/SlowFast-Interaction/outputs/interaction/2022/{ww}/slowfast/{approach}/{filename}/grad_cam/Path_0/{label_name}",
               f"/media/kiyoaki/Ventoy/projects/SlowFast-Interaction/outputs/interaction/2022/{ww}/slowfast/{approach}/{filename}/grad_cam/")