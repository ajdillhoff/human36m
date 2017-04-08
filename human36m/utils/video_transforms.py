import torch
import random
from PIL import Image
import numpy as np
import numbers

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor()
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq):
        for t in self.transforms:
            seq = t(seq)
        return seq

class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B), this normalizes each channel
    of every image in the given sequence, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, seq):
        for i in range(seq.size(0)):
            for t, m, s in zip(seq[i], self.mean, self.std):
                t.sub_(m).div_(s)
        return seq

class ToTensor(object):
    """Converts a numpy.ndarray (N x H x W x C) in the range [0, 255] to a 
    torch.FloatTensor of shape (N x C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, seq):
        seq = torch.FloatTensor(seq.transpose((0, 3, 1, 2)).astype(np.float32))
        for i in range(seq.size(0)):
            img = seq[i]
            img = img.float().div(255)
            seq[i] = img
        return seq

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the images in the given ndarray with a
    probability of 0.5
    """

    def __call__(self, seq):
        if random.random() < 0.5:
            for i in range(seq.shape[0]):
                img = Image.fromarray(seq[i].astype(np.uint8))
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                seq[i] = np.array(img)
            return seq
        return seq

class RandomCrop(object):
    """Crops the given video sequence at a random location to have a region of
    the given size. The temporal dimension is also cropped. Assumes an input
    sequence of shape (N x H x W x C).
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size(int(size), int(size), int(size))
        else:
            self.size = size

    def __call__(self, seq):
        h = seq.shape[1]
        w = seq.shape[2]
        n = seq.shape[0]
        tn, th, tw = self.size

        if h == th and w == tw and n == tn:
            return seq

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        n1 = random.randint(0, n - tn)

        new_seq = np.zeros((tn, th, tw, seq.shape[3]))
        new_seq_idx = 0

        for i in range(n1, n1 + tn):
            img = Image.fromarray(seq[i].astype(np.uint8))
            img = img.crop((x1, y1, x1 + tw, y1 + th))
            new_seq[new_seq_idx] = np.array(img)

        return new_seq
