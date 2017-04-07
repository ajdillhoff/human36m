import human36m.utils.video_utils as video_utils
import numpy as np
import matplotlib.pyplot as plt

def imshow(inp):
    """imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    print(inp.shape)

    # Image originally loaded in BGR, need to convert to RGB
    img = inp.copy()
    #img[:, :, 0] = inp[:, :, 2]
    #img[:, :, 2] = inp[:, :, 0]
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    #inp = std * inp + mean
    plt.imshow(img)

frame_idx = 0

path = "/media/adillhoff/Data Set 01/human3.6m/train/S1/Videos/Directions.55011271.mp4"
print("Loading video...")
video = video_utils.load_video(path, 40, np.float32)
#print("Resizing video...")
#video = video_utils.resize_video(video, (40, 3, 64, 64))
print("Visualizing frame {}".format(frame_idx))
frame = video[frame_idx, :, :, :]
print(frame.size())
imshow(frame)
plt.show()
