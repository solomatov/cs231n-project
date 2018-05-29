import torch
import torch.nn as nn
import torch.nn.functional as F

from dogsgan.data.dogs import create_dogs_dataset
from dogsgan.training.optimizers import VanillayGANOptimizer
from dogsgan.training.runner import TrainingRunner, BaseGenerator

ALPHA = 0.2
BATCH_SIZE = 128
NOISE_DIM = 1024
BASE_DIM = 128
WEIGHT_STD = 0.02
BASE_LR_128 = 0.0002
GRADIENT_CLIP = 10.0


def lrelu(x):
    return F.leaky_relu(x, ALPHA)


def init_weight(layer):
    layer.weight.data.normal_(0.0, WEIGHT_STD)


def init_weights(module):
    for c in module.children():
        init_weight(c)


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

        init_weights(self)

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
        self.bn1 = nn.BatchNorm2d(base)
        self.conv2 = nn.Conv2d(base, base * 2, (5, 5), padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(base * 2)
        self.conv3 = nn.Conv2d(base * 2, base * 4, (5, 5), padding=2, stride=2)
        self.bn3 = nn.BatchNorm2d(base * 4)
        self.conv4 = nn.Conv2d(base * 4, base * 8, (5, 5), padding=2, stride=2)
        self.bn4 = nn.BatchNorm2d(base * 8)
        self.collapse = nn.Linear((base * 8) * 4 * 4, 1)

        init_weights(self)

    def forward(self, x):
        z1 = lrelu(self.bn1(self.conv1(x)))
        z2 = lrelu(self.bn2(self.conv2(z1)))
        z3 = lrelu(self.bn3(self.conv3(z2)))
        z4 = lrelu(self.bn4(self.conv4(z3)))
        z5 = self.collapse(z4.view(-1, (self.base * 8) * 4 * 4))
        return F.sigmoid(z5)


if __name__ == '__main__':
    opt = VanillayGANOptimizer()
    runner = TrainingRunner(
        'dcgan', create_dogs_dataset(),
        Generator(), Discriminator(),
        VanillayGANOptimizer(dsc_lr=BASE_LR_128, gen_lr=BASE_LR_128))

    runner.run(batch_size=BATCH_SIZE)