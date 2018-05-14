import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from dogsgan.data.loader import create_loader

ALPHA = 0.02
NOISE_DIM = 1024
LABEL_NOISE = 0.4


def lrelu(x):
    return F.leaky_relu(x, ALPHA)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.base = 128
        base = self.base

        self.noise_project = nn.Linear(NOISE_DIM, 4 * 4 * base * 8)
        self.conv1 = nn.ConvTranspose2d(base * 8, base * 4, (4, 4), stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(base * 4, base * 2, (4, 4), stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(base * 2, base, (4, 4), stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(base, 3, (4, 4), stride=2, padding=1)

    def forward(self, z):
        z0 = self.noise_project(z).view(-1, self.base * 8, 4, 4)
        z1 = lrelu(self.conv1(z0))
        z2 = lrelu(self.conv2(z1))
        z3 = lrelu(self.conv3(z2))
        z4 = lrelu(self.conv4(z3))
        return F.tanh(z4)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.base = 128
        base = self.base

        self.conv1 = nn.Conv2d(3, base, (4, 4), padding=1, stride=2)
        self.conv2 = nn.Conv2d(base, base * 2, (4, 4), padding=1, stride=2)
        self.conv3 = nn.Conv2d(base * 2, base * 4, (4, 4), padding=1, stride=2)
        self.conv4 = nn.Conv2d(base * 4, base * 8, (4, 4), padding=1, stride=2)
        self.collapse = nn.Linear((base * 8) * 4 * 4, 1)

    def forward(self, x):
        z1 = lrelu(self.conv1(x))
        z2 = lrelu(self.conv2(z1))
        z3 = lrelu(self.conv3(z2))
        z4 = lrelu(self.conv4(z3))
        return F.sigmoid(self.collapse(z4.view(-1, (self.base * 8) * 4 * 4)))


if __name__ == '__main__':
    has_cuda = torch.cuda.device_count() > 0
    if has_cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu:0')

    gen = Generator().to(device)
    dsc = Discriminator().to(device)

    loader = create_loader()

    dsc_opt = optim.Adam(dsc.parameters())
    gen_opt = optim.Adam(gen.parameters())

    for e in range(1000):
        for i, (X_real, _) in enumerate(loader):
            dsc.zero_grad()

            N = X_real.shape[0]
            X_real = X_real.to(device)
            X_fake = gen(torch.randn((N, NOISE_DIM)).to(device))
            y_fake = (torch.rand((N, 1)) * LABEL_NOISE).to(device)
            y_real = (1.0 - torch.rand((N, 1)) * LABEL_NOISE).to(device)

            X = torch.cat([X_real, X_fake])
            y = torch.cat([y_real, y_fake])
            y_ = dsc(X)

            dsc_loss = F.binary_cross_entropy(y_, y)
            dsc_loss.backward()
            dsc_opt.step()

            dsc.zero_grad()
            gen.zero_grad()
            X_fake = gen(torch.randn((N * 2, NOISE_DIM)).to(device))
            y_ = dsc(X_fake)

            gen_loss = F.binary_cross_entropy(y_, (1.0 - torch.rand((N * 2, 1)) * (LABEL_NOISE / 2)).to(device))
            gen_loss.backward()
            gen_opt.step()

            if i % 10 == 0:
                print(f'{e:04}:{i:04}: Losses: dsc = {dsc_loss:.4f}, gen = {gen_loss:.4f}')

        print('saving sample...')
        sample = gen(torch.randn((100, NOISE_DIM)).to(device))
        torchvision.utils.save_image(sample, f'out/out-{e:04}.png', normalize=True)
        print('done')