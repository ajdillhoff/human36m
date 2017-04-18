import torch
import torch.nn as nn
import torch.nn.functional as F
import model
from torch.autograd import Variable

N = 10
C_in = 3
D_in = 16
H = 224
W = 224
num_classes = 10

if __name__ == "__main__":
    m = model.AlexNet(32)

    x = Variable(torch.randn(N, C_in, H, W))
    class_weights = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    # y = Variable(torch.multinomial(class_weights, N, replacement=True),
    #         requires_grad=False)
    y = Variable(torch.rand(N, 64))

    loss_fn = torch.nn.MSELoss()

    y_pred = m.forward(x)
    print(y_pred.size())

    loss = loss_fn(y_pred, y)
    print(loss)
