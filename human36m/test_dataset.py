from PIL import Image, ImageDraw
from timeit import default_timer as timer
from utils import data_transforms
import numpy as np
import human36m
import os

def draw_skeleton(img, joints, size, color):
    draw = ImageDraw.Draw(img)
    for i in range(joints.shape[0]):
        s = (joints[i, 0] - size, joints[i, 1] - size)
        t = (joints[i, 0] + size, joints[i, 1] + size)
        draw.ellipse([s, t], fill=color)

    img.show()

img_path = os.path.normpath("/media/adillhoff/Data Set 01/human3.6m/images")
target_path = os.path.normpath("/media/adillhoff/Data Set 01/human3.6m/train")
target_post = "/MyPoseFeatures/D2_Positions"

dset = human36m.HUMAN36MPose(img_path, target_path,
        transform=data_transforms.Compose([
            data_transforms.CropToTarget(20),
            data_transforms.Scale((220, 220)),
            data_transforms.RandomHorizontalFlip(),
            data_transforms.ToTensor(),
        ]))

print(len(dset))

for i in range(128):
    start = timer()
    index = np.random.randint(len(dset))
    img, target = dset.__getitem__(index)
    end = timer()
    print(end - start)
# print(target.reshape(32, 2))

# draw_skeleton(img, target.reshape(32, 2), 5, "#a00000")
