import os

from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from spacepy import pycdf
import h5py

class HUMAN36MVideo(data.Dataset):
    train_list = ["human36m_video.hdf5"]

    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.train_data = []
        self.train_labels = []
        for fentry in self.train_list:
            path = os.path.join(root, fentry)
            f = h5py.File(path, 'r')
            self.train_data = f['.']['data']
            self.train_labels = f['.']['labels']
            # f.close()

        self.num_frames = self.train_data.shape[1]
        self.img_height = self.train_data.shape[2]
        self.img_width = self.train_data.shape[3]
        self.channels = self.train_data.shape[4]

    def __getitem__(self, index):
        if self.train:
            seq, target = self.train_data[index], self.train_labels[index]

        if self.transform is not None:
            seq = self.transform(seq)
            # The sequence should be (num_frames, channels, H, W)
            # sample = torch.Tensor(self.num_frames, self.channels,
            #         self.img_height, self.img_width)
            # for seq_index in range(seq.shape[0]):
            #     img = Image.fromarray(seq[seq_index].astype(np.uint8))
            #     img = self.transform(img)
            #     sample[seq_index] = img

            # For 3DCNN, we need a volume that is (C, D, H, W)
            seq = np.transpose(seq.numpy(), (1, 0, 2, 3))
            seq = torch.Tensor(seq)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # h5py loads numpy objects, the labels need to be converted to scalars
        target = target.item()

        return seq, target

    def __len__(self):
        if self.train:
            return self.train_data.shape[0]

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG",
    ".png", ".PNG", ".bmp", ".BMP",
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(img_dir, target_dir):
    images = []
    target_buffer = {}
    targets = []

    # load images
    for subject in os.listdir(img_dir):
        d = os.path.join(img_dir, subject)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    video_name = fname.split("_")[0]
                    index = fname.split("_")[1].split(".")[0]
                    index = int(index)
                    target_key = subject + video_name

                    if not (target_key in target_buffer):
                        t = load_targets(target_dir, fname, subject)
                        target_buffer[target_key] = t
                    target = target_buffer[target_key]

                    if index < target.shape[0]:
                        images.append(path)
                        targets.append(target[index, :])

    return images, targets

def load_targets(target_path, file_name, subject):
    target_prefix = "/MyPoseFeatures/D2_Positions"
    target_path = os.path.join(target_path, subject + target_prefix)
    file_meta = file_name.split("_")
    activity = file_meta[0]
    cdf = pycdf.CDF(os.path.join(target_path, activity + ".cdf"))
    targets = cdf[0]
    targets = targets[0, :, :]
    cdf.close()
    return targets

def default_loader(path):
    return Image.open(path).convert("RGB")

class HUMAN36MPose(data.Dataset):
    def __init__(self, base_path, target_path, transform=None):
        imgs, targets = make_dataset(base_path, target_path)
        self.imgs = imgs
        self.targets = targets
        self.base_path = base_path
        self.target_path = target_path
        self.transform = transform
        self.loader = default_loader

    def target_loader(self, path, index):
        file_name = path.split("/")[-1]
        subject = path.split("/")[-2]

        if file_name in self.targets:
            targets = self.targets[file_name]
        else:
            targets = load_targets(self.target_path, file_name, subject)
            self.targets[file_name] = targets

        if index >= targets.shape[0]:
            target = targets[-1, :]
        else:
            target = targets[index, :]

        return target

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        target = self.targets[index]

        if self.transform is not None:
            target = target.reshape(-1, 2)
            img, target = self.transform(img, target)
            target = target.reshape(-1)

        return img, target

    def __len__(self):
        return len(self.imgs)

