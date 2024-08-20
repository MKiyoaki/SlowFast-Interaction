from helper.configurations import output_dir
from helper.video_helper import combine_videos, combine_frames


def combine_output():
    yyyy = "2022"
    ww = "w1"



    combine_videos(f"{output_dir}/interaction/{yyyy}/{ww}/slowfast/ua/1_w1234_ft/grad_cam/Path_1/InteractionRupture",
                  f"{output_dir}/interaction/{yyyy}/{ww}/slowfast/ua/1_w1234_ft/grad_cam/")

    combine_frames(f"{output_dir}/interaction/{yyyy}/{ww}/slowfast/ua/1_w1234_sampling_ft/grad_cam/Path_0/RobotMistake",
                   f"{output_dir}/interaction/{yyyy}/{ww}/slowfast/ua/1_w1234_sampling_ft/grad_cam/")


if __name__ == "__main__":
    combine_output()
