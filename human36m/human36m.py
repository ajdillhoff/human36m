import os

from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
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

class HUMAN36MPose(data.Dataset):
    train_list = ["human36m_pose.hdf5"]

    def __init__(self, file_root, data_root, transform=None, 
            target_transform=None):
        self.file_root = file_root
        self.data_root = data_root
        self.transform = transform
        self.target_transform = target_transform

        self.data = []
        self.targets = []
        for fentry in self.train_list:
            path = os.path.join(file_root, fentry)
            f = h5py.File(path, 'r')
            self.data = f[data_root]['data']
            self.targets = f[data_root]['targets']
            # f.close()

        self.img_height = self.data.shape[1]
        self.img_width = self.data.shape[2]
        self.channels = self.data.shape[3]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.astype(np.uint8))

        if self.transform is not None:
            target = target.reshape(-1, 2)
            img, target = self.transform(img, target)
            target = target.reshape(-1)

        return img, target

    def __len__(self):
        return self.data.shape[0]
