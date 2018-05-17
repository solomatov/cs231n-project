import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from dogsgan.data.loader import create_loader

ALPHA = 0.2
BATCH_SIZE = 128
NOISE_DIM = 100
LABEL_NOISE = 0.00
BASE_DIM = 128
WEIGHT_STD = 0.02
BASE_LR_128 = 0.0002

ZERO_LABEL_MEAN = 2 * LABEL_NOISE
ONE_LABEL_MEAN = 1.0 - 2 * LABEL_NOISE


def lrelu(x):
    return F.leaky_relu(x, ALPHA)


def init_weight(layer):
    layer.weight.data.normal_(0.0, WEIGHT_STD)


def init_weights(module):
    for c in module.children():
        init_weight(c)


def gen_labels(shape, mean):
    result = torch.randn(shape) * WEIGHT_STD + mean
    return result.clamp(min=1e-7, max=1.0)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.base = BASE_DIM
        base = self.base

        self.noise_project = nn.Linear(NOISE_DIM, 4 * 4 * base * 8)
        self.conv1 = nn.ConvTranspose2d(base * 8, base * 4, (5, 5), stride=2, padding=2, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(base * 4, base * 2, (5, 5), stride=2, padding=2, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(base * 2, base, (5, 5), stride=2, padding=2, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(base, 3, (5, 5), stride=2, padding=2, output_padding=1)

        init_weights(self)

    def forward(self, z):
        z0 = lrelu(self.noise_project(z).view(-1, self.base * 8, 4, 4))
        z1 = lrelu(self.conv1(z0))
        z2 = lrelu(self.conv2(z1))
        z3 = lrelu(self.conv3(z2))
        z4 = lrelu(self.conv4(z3))
        return F.tanh(z4)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.base = BASE_DIM
        base = self.base

        self.conv1 = nn.Conv2d(3, base, (5, 5), padding=2, stride=2)
        self.conv2 = nn.Conv2d(base, base * 2, (5, 5), padding=2, stride=2)
        self.conv3 = nn.Conv2d(base * 2, base * 4, (5, 5), padding=2, stride=2)
        self.conv4 = nn.Conv2d(base * 4, base * 8, (5, 5), padding=2, stride=2)
        self.collapse = nn.Linear((base * 8) * 4 * 4, 1)

        init_weights(self)

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

    loader = create_loader(batch_size=BATCH_SIZE)

    scaled_lr = BASE_LR_128 * (BATCH_SIZE / 128)

    dsc_opt = optim.Adam(dsc.parameters(), lr=scaled_lr / 8, betas=(0.5, 0.999))
    gen_opt = optim.Adam(gen.parameters(), lr=scaled_lr, betas=(0.5, 0.999))

    vis_params = torch.randn((104, NOISE_DIM)).to(device)

    for e in range(1000):
        for i, (X_real, _) in enumerate(loader):
            dsc.zero_grad()

            N = X_real.shape[0]
            X_real = X_real.to(device)
            X_fake = gen(torch.randn((N, NOISE_DIM)).to(device))
            y_fake = gen_labels((N, 1), ZERO_LABEL_MEAN).to(device)
            y_real = gen_labels((N, 1), ONE_LABEL_MEAN).to(device)

            X = torch.cat([X_real, X_fake])
            y = torch.cat([y_real, y_fake])
            y_ = dsc(X)

            dsc_loss = F.binary_cross_entropy(y_, y)
            dsc_loss.backward()
            dsc_opt.step()

            dsc.zero_grad()
            gen.zero_grad()
            X_fake = gen(torch.randn((N, NOISE_DIM)).to(device))
            y_ = dsc(X_fake)

            gen_loss = -torch.mean(torch.log(y_))
            gen_loss.backward()
            gen_opt.step()

            if i % 10 == 0:
                print(f'{e:04}:{i:04}: Losses: dsc = {dsc_loss:.4f}, gen = {gen_loss:.4f}')

        print('saving sample...')
        sample = gen(vis_params)
        torchvision.utils.save_image(sample, f'out/out-{e:04}.png', normalize=True)
        print('done')