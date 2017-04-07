import human36m
import model
import torch.optim as optim
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable

num_epochs = 10
learning_rate = 5e-4
momentum = 0.9
torch.cuda.manual_seed(1)

m = model.Model()
m.cuda()
m.train()

optimizer = optim.SGD(m.parameters(), lr=learning_rate, momentum=momentum)

print("Loading data...")
a = human36m.HUMAN36M("", transform=transforms.Compose([
        transforms.ToTensor()
    ]))
print("Starting \"training\"")

train_loader = torch.utils.data.DataLoader(a, batch_size=2, shuffle=True)

for epoch in range(0, num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda()
        target = target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = m(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
