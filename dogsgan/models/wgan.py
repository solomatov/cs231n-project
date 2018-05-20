import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from dogsgan.data.loader import create_loader

ALPHA = 0.2
BATCH_SIZE = 64
N_CRITIC = 5
CLIP = 0.01
LEARNING_RATE = 0.00005
NOISE_DIM = 100
BASE_DIM = 128
WEIGHT_STD = 0.02


def lrelu(x):
    return F.leaky_relu(x, ALPHA)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.base = BASE_DIM
        base = self.base

        self.noise_project = nn.Linear(NOISE_DIM, 4 * 4 * base * 8)
        self.conv1 = nn.ConvTranspose2d(base * 8, base * 4, (5, 5), stride=2, padding=2, output_padding=1)
        self.bn1 = nn.BatchNorm2d(base * 4)
        self.conv2 = nn.ConvTranspose2d(base * 4, base * 2, (5, 5), stride=2, padding=2, output_padding=1)
        self.bn2 = nn.BatchNorm2d(base * 2)
        self.conv3 = nn.ConvTranspose2d(base * 2, base, (5, 5), stride=2, padding=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(base)
        self.conv4 = nn.ConvTranspose2d(base, 3, (5, 5), stride=2, padding=2, output_padding=1)

    def forward(self, z):
        z0 = lrelu(self.noise_project(z).view(-1, self.base * 8, 4, 4))
        z1 = lrelu(self.bn1(self.conv1(z0)))
        z2 = lrelu(self.bn2(self.conv2(z1)))
        z3 = lrelu(self.bn3(self.conv3(z2)))
        z4 = lrelu(self.conv4(z3))
        return F.tanh(z4)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.base = BASE_DIM
        base = self.base

        self.conv1 = nn.Conv2d(3, base, (5, 5), padding=2, stride=2)
        self.bn1 = nn.BatchNorm2d(base)
        self.conv2 = nn.Conv2d(base, base * 2, (5, 5), padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(base * 2)
        self.conv3 = nn.Conv2d(base * 2, base * 4, (5, 5), padding=2, stride=2)
        self.bn3 = nn.BatchNorm2d(base * 4)
        self.conv4 = nn.Conv2d(base * 4, base * 8, (5, 5), padding=2, stride=2)
        self.bn4 = nn.BatchNorm2d(base * 8)
        self.collapse = nn.Linear((base * 8) * 4 * 4, 1)

    def forward(self, x):
        z1 = lrelu(self.bn1(self.conv1(x)))
        z2 = lrelu(self.bn2(self.conv2(z1)))
        z3 = lrelu(self.bn3(self.conv3(z2)))
        z4 = lrelu(self.bn4(self.conv4(z3)))
        return self.collapse(z4.view(-1, (self.base * 8) * 4 * 4))


if __name__ == '__main__':
    has_cuda = torch.cuda.device_count() > 0
    if has_cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu:0')

    gen = Generator().to(device)
    dsc = Discriminator().to(device)
    for p in dsc.parameters():
        p.data.clamp_(-CLIP, CLIP)

    loader = create_loader(batch_size=BATCH_SIZE)

    dsc_opt = optim.RMSprop(dsc.parameters(), lr=LEARNING_RATE)
    gen_opt = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)

    vis_params = torch.randn((104, NOISE_DIM)).to(device)
    for e in range(1000):

        it = iter(loader)

        i = 0
        while True:
            try:
                for _ in range(N_CRITIC):
                    X_real, _ = it.next()
                    dsc.zero_grad()
                    gen.zero_grad()
                    X_real = X_real.to(device)
                    X_fake = gen(torch.randn((X_real.shape[0], NOISE_DIM)).to(device))
                    dsc_f = dsc(X_fake)
                    dsc_r = dsc(X_real)
                    dsc_loss = torch.mean(dsc_f) - torch.mean(dsc_r)
                    dsc_loss.backward()
                    dsc_opt.step()

                    for p in dsc.parameters():
                        p.data.clamp_(-CLIP, CLIP)

                dsc.zero_grad()
                gen.zero_grad()
                noise = torch.randn((BATCH_SIZE, NOISE_DIM)).to(device)
                gen_loss = -torch.mean(dsc(gen(noise)))
                gen_loss.backward()
                gen_opt.step()

                if i % 10 == 0:
                    print(f'{e:04}:{i:04}: Losses: dsc = {dsc_loss:.4f}, gen = {gen_loss:.4f}')

                i += 1
            except StopIteration:
                break


        print('saving sample...')
        sample = gen(vis_params)
        torchvision.utils.save_image(sample, f'out/out-{e:04}.png', normalize=True)
        print('done')