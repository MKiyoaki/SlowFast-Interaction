from pytorchvideo.data import LabeledVideoDataset

"""
TODO: ADD THE DOCUMENTATION
"""


class InteractionDataset(LabeledVideoDataset):
    def __init__(self, labeled_video_paths, clip_sampler, decode_audio, decode_video, transform=None):
        super().__init__(
            labeled_video_paths,
            clip_sampler,
            decode_audio=decode_audio,
            decode_video=decode_video,
            transform=transform
        )
