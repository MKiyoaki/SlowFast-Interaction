import math
import os
import subprocess

import cv2
from PIL import Image


def combine_videos(target_dir, output_dir=None):
    """
    Combined the videos into a single video file.

    Args:
        target_dir (string)
        output_dir (string)
    """
    if output_dir is None:
        output_dir = target_dir

    videos = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    videos.sort()

    if not videos:
        print("No video found")
        return

    list_file = os.path.join(target_dir, 'videos.txt')
    with open(list_file, 'w') as f:
        for video in videos:
            f.write(f"file '{video}'\n")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, 'combined_video.mp4')

    command = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', list_file,
        '-c', 'copy',
        output_file
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Successfully combine videos: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to combine videos: {e}")
    finally:
        os.remove(list_file)


def combine_frames(target_dir, output_dir=None):
    """
    Combines the frames into a single video file.

    Args:
        target_dir (string)
        output_dir (string)
    """
    if output_dir is None:
        output_dir = target_dir

    videos = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if
              f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    videos.sort()

    if not videos:
        print("No video found")
        return

    frames = []
    for video in videos:
        frames_from_video = extract_frames_from_video(video, 1)
        frames.extend(frames_from_video)

    if not frames:
        print("No frames extracted")
        return

    num_frames = len(frames)
    grid_size = math.ceil(math.sqrt(num_frames))
    frame_height, frame_width, _ = frames[0].shape
    grid_image = Image.new('RGB', (grid_size * frame_width, grid_size * frame_height))

    for idx, frame in enumerate(frames):
        row = idx // grid_size
        col = idx % grid_size
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        grid_image.paste(pil_img, (col * frame_width, row * frame_height))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, 'frame_grid.jpg')
    grid_image.save(output_file)
    print(f"Successfully created frame grid: {output_file}")


def extract_frames_from_video(video_dir, frame_count=1):
    cap = cv2.VideoCapture(video_dir)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // frame_count, 1)

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        if len(frames) >= frame_count:
            break

    cap.release()
    return frames


def convert_video_to_frame(video_dir, frame_count=1):
    """
    从指定目录中的所有视频中提取帧，并将帧保存到平行目录的 imgs 文件夹中。

    参数:
    - video_dir (str): 包含视频文件的目录路径。
    - frame_count (int): 要从每个视频中提取的帧数。

    返回:
    - frames (list): 提取的帧图像列表。
    """

    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(video_dir), 'imgs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frames = []  # 存储所有视频提取的帧

    # 遍历目录中的所有视频文件
    for video in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video)
        video_name = os.path.splitext(video)[0]  # 获取视频文件名（不包括扩展名）

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(total_frames // frame_count, 1)

        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                # 生成每个帧的唯一文件名
                frame_filename = f"{video_name}_frame_{i}.png"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)

            if len(frames) >= frame_count:
                break

        cap.release()

    return frames
