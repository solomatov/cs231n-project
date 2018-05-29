import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from dogsgan.training.runner import TrainingRunner, BaseGenerator
from dogsgan.training.optimizers import WGANGPOptimizer
from dogsgan.data.dogs import create_dogs_dataset

ALPHA = 0.2
BATCH_SIZE = 64
N_CRITIC = 5
CLIP = 0.01
LEARNING_RATE = 1e-4
NOISE_DIM = 1024
BASE_DIM = 128
WEIGHT_STD = 0.02
LAMBDA = 10.0


def lrelu(x):
    return F.leaky_relu(x, ALPHA)


def grad_norm(m):
    return max(p.abs().max() for p in m.parameters())


def init_modules(root):
    for m in root.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, WEIGHT_STD)


class Generator(BaseGenerator):
    def __init__(self):
        super().__init__()

        self.base = BASE_DIM
        base = self.base

        self.noise_project = nn.Linear(NOISE_DIM, 4 * 4 * base * 8)
        self.bn0 = nn.BatchNorm1d(4 * 4 * base * 8)
        self.conv1 = nn.ConvTranspose2d(base * 8, base * 4, (5, 5), stride=2, padding=2, output_padding=1)
        self.bn1 = nn.BatchNorm2d(base * 4)
        self.conv2 = nn.ConvTranspose2d(base * 4, base * 2, (5, 5), stride=2, padding=2, output_padding=1)
        self.bn2 = nn.BatchNorm2d(base * 2)
        self.conv3 = nn.ConvTranspose2d(base * 2, base, (5, 5), stride=2, padding=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(base)
        self.conv4 = nn.ConvTranspose2d(base, 3, (5, 5), stride=2, padding=2, output_padding=1)

        init_modules(self)

    def forward(self, z):
        z0 = lrelu(self.bn0(self.noise_project(z)).view(-1, self.base * 8, 4, 4))
        z1 = lrelu(self.bn1(self.conv1(z0)))
        z2 = lrelu(self.bn2(self.conv2(z1)))
        z3 = lrelu(self.bn3(self.conv3(z2)))
        z4 = self.conv4(z3)
        return F.tanh(z4)

    def gen_noise(self, n):
        return torch.randn((n, NOISE_DIM), device=self.get_device())


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.base = BASE_DIM
        base = self.base

        self.conv1 = nn.Conv2d(3, base, (5, 5), padding=2, stride=2)
        self.ln1 = nn.LayerNorm([base, 32, 32])
        self.conv2 = nn.Conv2d(base, base * 2, (5, 5), padding=2, stride=2)
        self.ln2 = nn.LayerNorm([base * 2, 16, 16])
        self.conv3 = nn.Conv2d(base * 2, base * 4, (5, 5), padding=2, stride=2)
        self.ln3 = nn.LayerNorm([base * 4, 8, 8])
        self.conv4 = nn.Conv2d(base * 4, base * 8, (5, 5), padding=2, stride=2)
        self.ln4 = nn.LayerNorm([base * 8, 4, 4])
        self.collapse = nn.Linear((base * 8) * 4 * 4, 1)


        init_modules(self)

    def forward(self, x):
        z1 = lrelu(self.ln1(self.conv1(x)))
        z2 = lrelu(self.ln2(self.conv2(z1)))
        z3 = lrelu(self.ln3(self.conv3(z2)))
        z4 = lrelu(self.ln4(self.conv4(z3)))
        return self.collapse(z4.view(-1, (self.base * 8) * 4 * 4))


if __name__ == '__main__':
    runner = TrainingRunner('wgan_gp', create_dogs_dataset(), Generator(), Discriminator(), WGANGPOptimizer())
    runner.run(batch_size=BATCH_SIZE)