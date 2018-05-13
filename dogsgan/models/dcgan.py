import torch
import torch.nn as nn
import torch.nn.functional as F


ALPHA = 0.02


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.noise_project = nn.Linear(100, 4 * 4 * 1024)
        self.conv1 = nn.ConvTranspose2d(1024, 512, (4, 4), stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(512, 256, (4, 4), stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(256, 128, (4, 4), stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(128, 3, (4, 4), stride=2, padding=1)

    def forward(self, z):
        z0 = self.noise_project(z).view(-1, 1024, 4, 4)
        z1 = F.leaky_relu(self.conv1(z0), ALPHA)
        z2 = F.leaky_relu(self.conv2(z1), ALPHA)
        z3 = F.leaky_relu(self.conv3(z2), ALPHA)
        z4 = F.leaky_relu(self.conv4(z3), ALPHA)
        return F.tanh(z4)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 128, (4, 4), padding=1, stride=2)
        self.conv2 = nn.Conv2d(128, 256, (4, 4), padding=1, stride=2)
        self.conv3 = nn.Conv2d(256, 512, (4, 4), padding=1, stride=2)
        self.conv4 = nn.Conv2d(512, 1024, (4, 4), padding=1, stride=2)
        self.collapse = nn.Linear(1024 * 4 * 4, 1)

    def forward(self, x):
        z1 = F.relu(self.conv1(x))
        z2 = F.relu(self.conv2(z1))
        z3 = F.relu(self.conv3(z2))
        z4 = F.relu(self.conv4(z3))
        return F.sigmoid(self.collapse(z4.view(-1, 1024 * 4 * 4)))


if __name__ == '__main__':
    gen = Generator()
    dsc = Discriminator()

    print(gen(torch.empty(1, 100).uniform_(0, 1)).shape)
    print(dsc(torch.empty(1, 3, 64, 64).uniform_(0, 1)).shape)

