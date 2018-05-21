import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from dogsgan.infra.runner import TrainingRunner


ALPHA = 0.2
BATCH_SIZE = 128
NOISE_DIM = 100
LABEL_NOISE = 0.00
BASE_DIM = 128
WEIGHT_STD = 0.02
BASE_LR_128 = 0.0002
GRADIENT_CLIP = 10.0

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


class DCGANRunner(TrainingRunner):
    def __init__(self):
        super().__init__('dcgan')
        self.gen = Generator().to(self.device)
        self.dsc = Discriminator().to(self.device)

        scaled_lr = BASE_LR_128 * (BATCH_SIZE / 128)
        self.dsc_opt = optim.Adam(self.dsc.parameters(), lr=scaled_lr / 6, betas=(0.5, 0.999))
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=scaled_lr, betas=(0.5, 0.999))

        self.vis_params = torch.randn((104, NOISE_DIM)).to(self.device)

    def run_epoch(self, it, context):
        for X_real, _ in it:
            self.dsc.zero_grad()

            N = X_real.shape[0]
            X_real = X_real.to(self.device)
            X_fake = self.gen(torch.randn((N, NOISE_DIM)).to(self.device))
            y_fake = gen_labels((N, 1), ZERO_LABEL_MEAN).to(self.device)
            y_real = gen_labels((N, 1), ONE_LABEL_MEAN).to(self.device)

            X = torch.cat([X_real, X_fake])
            y = torch.cat([y_real, y_fake])
            y_ = self.dsc(X)

            dsc_loss = F.binary_cross_entropy(y_, y)
            dsc_loss.backward()
            self.dsc_opt.step()

            self.dsc.zero_grad()
            self.gen.zero_grad()
            X_fake = self.gen(torch.randn((N, NOISE_DIM)).to(self.device))
            y_ = self.dsc(X_fake)

            gen_loss = -torch.mean(torch.log(y_))
            gen_loss.backward()
            self.gen_opt.step()

    def sample_images(self):
        return self.gen(self.vis_params)

    def get_snapshot(self):
        return {
            'gen': self.gen,
            'dsc': self.dsc
        }


if __name__ == '__main__':
    runner = DCGANRunner()
    runner.run(batch_size=BATCH_SIZE)