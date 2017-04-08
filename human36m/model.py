import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv3d(3, 16, (3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(16, 32, (3, 3, 3), padding=1)
        self.conv3 = nn.Conv3d(32, 64, (3, 3, 3), padding=1)
        self.conv4 = nn.Conv3d(64, 128, (3, 3, 3), padding=1)
        self.conv5 = nn.Conv3d(128, 128, (3, 3, 3), padding=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 15)

    def forward(self, x):
        x = F.relu(F.max_pool3d(self.conv1(x), (1, 2, 2)))
        x = F.relu(F.max_pool3d(self.conv2(x), 2))
        x = F.relu(F.max_pool3d(self.conv3(x), 2))
        x = F.relu(F.max_pool3d(self.conv4(x), 2))
        x = F.relu(F.max_pool3d(self.conv5(x), 2))
        x = x.view(-1, 2048)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=True)
        x = F.relu(self.fc2(x))
        return x
