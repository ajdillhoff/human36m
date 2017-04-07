from unittest import TestCase

import numpy as np
import human36m.utils.video_utils as video_utils

def run_tests():
    TestVideoUtils.test()

class TestVideoUtils(TestCase):

    def test(self):
        path = "/media/adillhoff/Data Set 01/human3.6m/train/S1/Videos/Directions.55011271.mp4"

        num_frames = 40
        num_channels = 3
        height = 64
        width = 64

        video = video_utils.load_video(path, np.float32)
        video = video_utils.resize_video(video,
                (num_frames, num_channels, height, width))

        self.assertTrue(video.size() == (num_frames, num_channels, height, width))

if __name__ == '__main__':
    run_tests()
