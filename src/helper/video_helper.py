import os
import subprocess


def combine_videos(target_dir, output_dir):
    videos = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    videos.sort()

    if not videos:
        print("No video found")
        return

    list_file = os.path.join(target_dir, 'videos.txt')
    with open(list_file, 'w') as f:
        for video in videos:
            f.write(f"file '{video}'\n")

    output_file = os.path.join(output_dir, 'combined_video.mp4')

    command = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', list_file,
        '-c', 'copy',
        output_file
    ]

    # 执行命令
    try:
        subprocess.run(command, check=True)
        print(f"视频合并完成，输出文件为: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"视频合并失败: {e}")
    finally:
        # 删除临时文件
        os.remove(list_file)
