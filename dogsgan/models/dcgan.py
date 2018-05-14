import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from dogsgan.data.loader import create_loader

ALPHA = 0.02
NOISE_DIM = 100


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.noise_project = nn.Linear(NOISE_DIM, 4 * 4 * 1024)
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

    loader = create_loader()

    dsc_opt = optim.Adam(dsc.parameters())
    gen_opt = optim.Adam(gen.parameters())

    for e in range(1000):
        for i, (X_real, _) in enumerate(loader):
            dsc.zero_grad()

            N = X_real.shape[0]
            X_fake = gen(torch.randn((N, NOISE_DIM)))
            y_fake = torch.zeros((N, 1))
            y_real = torch.ones((N, 1))

            X = torch.cat([X_real, X_fake])
            y = torch.cat([y_real, y_fake])
            y_ = dsc(X)

            dsc_loss = F.binary_cross_entropy(y_, y)
            print(f'dsc loss = {dsc_loss:.2f}')

            dsc_loss.backward()
            dsc_opt.step()

            gen.zero_grad()
            X_fake = gen(torch.randn((N, NOISE_DIM)))
            y_ = dsc(X_fake)

            gen_loss = F.binary_cross_entropy(y_, torch.ones((N, 1)))
            print(f'gen loss = {gen_loss:.2f}')

            gen_loss.backward()
            gen_opt.step()

            if i % 100 == 0:
                print('saving sample...')
                sample = gen(torch.randn((100, NOISE_DIM)))
                torchvision.utils.save_image(sample, f'out/out-{i}.png', normalize=True)
                print('done')