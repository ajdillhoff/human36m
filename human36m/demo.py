import sys
import numpy as np
import model as Model
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from timeit import default_timer as timer
from torch.autograd import Variable
from PIL import Image, ImageDraw, ImageFont

trained_model = "weights/human36m.pth.tar"
model = Model.AlexNet(32)
weights = torch.load(trained_model, map_location=lambda storage, loc: storage)
model.load_state_dict(weights["state_dict"])

normalize = transforms.Normalize(
                mean=[0.00094127, 0.00060294, 0.0005603],
                std=[0.02102633, 0.01346872, 0.01251619]
            )

transform = transforms.Compose([transforms.Scale((220, 220)), 
                                transforms.ToTensor(),
                                normalize])

joint_names = [
            "Hips",
            "RightUpLeg",
            "RightLeg",
            "RightFoot",
            "RightToeBase",
            "Site",
            "LeftUpLeg",
            "LeftLeg",
            "LeftFoot",
            "LeftToeBase",
            "Site",
            "Spine",
            "Spine1",
            "Neck",
            "Head",
            "Site",
            "LeftShoulder",
            "LeftArm",
            "LeftForeArm",
            "LeftHand",
            "LeftHandThumb",
            "Site",
            "L_Wrist_End",
            "Site",
            "RightShoulder",
            "RightArm",
            "RightForeArm",
            "RightHand",
            "RightHandThumb",
            "Site",
            "R_Wrist_End",
            "Site"
        ]

joint_indices = [0, 3, 8, 14, 19, 27]

def draw_skeleton(img, joints, size):
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/Library/Fonts/Courier New.ttf", 16)
    for i in joint_indices:
        color = np.random.randint(255, size=(3, 1))
        s = (joints[i, 0] - size, joints[i, 1] - size)
        t = (joints[i, 0] + size, joints[i, 1] + size)
        draw.ellipse([s, t], fill=(color[0], color[1], color[2]))
        draw.text((s[0]-1, s[1]-1), joint_names[i], (0, 0, 0), font=font)
        draw.text((s[0]+1, s[1]-1), joint_names[i], (0, 0, 0), font=font)
        draw.text((s[0]-1, s[1]+1), joint_names[i], (0, 0, 0), font=font)
        draw.text((s[0]+1, s[1]+1), joint_names[i], (0, 0, 0), font=font)
        draw.text(s, joint_names[i], (255, 255, 255), font=font)
    return img

def predict(frame):
    x = transform(frame)
    x = Variable(transform(frame).unsqueeze(0))
    y = model(x)
    y = y.data.numpy()
    y = y.reshape(32, 2)
    w, h = frame.size
    y[:, 0] *= w
    y[:, 1] *= h

    frame = draw_skeleton(frame, y, 5)

    return frame

if __name__ == "__main__":
    file_name = sys.argv[1]
    img = Image.open(file_name)
    start = timer()
    img = predict(img)
    end = timer()
    print(end - start)
    img.show()
