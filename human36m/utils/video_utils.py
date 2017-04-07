import torch
import numpy as np
import os
import skvideo.io
import math

def load_video(file_path, num_frames, dtype=np.float32):
    """Reads a video and converts it to a Torch tensor with shape
       (num_frames, channels, height, width)
    """

    input_parameters = {"-pix_fmt": "rgb24"}

    reader = skvideo.io.FFmpegReader(file_path, outputdict=input_parameters)
    video_shape = reader.getShape()
    video_frames = video_shape[0]
    current_frame = 0

    video = np.zeros(video_shape)

    # Calculate the number of frames to skip in order to still get num_frames
    num_skip_frames = math.floor(video_frames / num_frames)

    for frame in reader.nextFrame():
        if current_frame % num_skip_frames == 0:
            video[current_frame, :, :, :] = frame

        current_frame += 1

    video = np.transpose(video, (0, 3, 1, 2))
    return torch.Tensor(video.astype(dtype))

def resize_video(video, video_size):
    """Resizes a given video using the given parameters.

    Arguments:
        video - torch.FloatTensor
        video_size - tuple (num_frames, channels, height, width)
    """

    video.resize_(video_size)
    return video
