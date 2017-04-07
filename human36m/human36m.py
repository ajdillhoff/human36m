from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
import h5py

class HUMAN36M(data.Dataset):
    # TODO: Take away the hardcoded path
    path = "/media/adillhoff/Data Set 01/human3.6m/human36m.hdf5"
    train_list = [path]

    def __init__(self, root, train=True,
            transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = h5py.File(fentry, 'r')
                self.train_data = f['.']['data'].value
                self.train_labels = f['.']['labels'].value
                f.close()

            self.num_frames = self.train_data.shape[1]
            self.img_height = self.train_data.shape[2]
            self.img_width = self.train_data.shape[3]
            self.channels = self.train_data.shape[4]

    def __getitem__(self, index):
        if self.train:
            seq, target = self.train_data[index], self.train_labels[index]

        if self.transform is not None:
            # The sequence should be (num_frames, channels, H, W)
            sample = torch.Tensor(self.num_frames, self.channels,
                    self.img_height, self.img_width)
            for seq_index in range(seq.shape[0]):
                img = Image.fromarray(seq[seq_index].astype(np.uint8))
                img = self.transform(img)
                sample[seq_index] = img

            # For 3DCNN, we need a volume that is (C, D, H, W)
            sample = np.transpose(sample.numpy(), (1, 0, 2, 3))
            seq = torch.Tensor(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # h5py loads numpy objects, the labels need to be converted to scalars
        target = target.item()

        return seq, target

    def __len__(self):
        if self.train:
            return self.train_data.shape[0]
