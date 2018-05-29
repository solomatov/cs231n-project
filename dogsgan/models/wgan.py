import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from dogsgan.infra.runner import TrainingRunner
from dogsgan.data.dogs import create_dogs_dataset

ALPHA = 0.2
BATCH_SIZE = 64
N_CRITIC = 5
CLIP = 0.01
LEARNING_RATE = 0.00005
NOISE_DIM = 1024
BASE_DIM = 128
WEIGHT_STD = 0.02


def lrelu(x):
    return F.leaky_relu(x, ALPHA)


def grad_norm(m):
    return max(p.abs().max() for p in m.parameters())


def init_modules(root):
    for m in root.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, WEIGHT_STD)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.base = BASE_DIM
        base = self.base

        self.noise_project = nn.Linear(NOISE_DIM, 4 * 4 * base * 8)
        self.bn0 = nn.BatchNorm1d(4 * 4 * base * 8, affine=True)
        self.conv1 = nn.ConvTranspose2d(base * 8, base * 4, (5, 5), stride=2, padding=2, output_padding=1)
        self.bn1 = nn.BatchNorm2d(base * 4, affine=True)
        self.conv2 = nn.ConvTranspose2d(base * 4, base * 2, (5, 5), stride=2, padding=2, output_padding=1)
        self.bn2 = nn.BatchNorm2d(base * 2, affine=True)
        self.conv3 = nn.ConvTranspose2d(base * 2, base, (5, 5), stride=2, padding=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(base, affine=True)
        self.conv4 = nn.ConvTranspose2d(base, 3, (5, 5), stride=2, padding=2, output_padding=1)

        init_modules(self)

    def forward(self, z):
        z0 = lrelu(self.bn0(self.noise_project(z)).view(-1, self.base * 8, 4, 4))
        z1 = lrelu(self.bn1(self.conv1(z0)))
        z2 = lrelu(self.bn2(self.conv2(z1)))
        z3 = lrelu(self.bn3(self.conv3(z2)))
        z4 = self.conv4(z3)
        return F.tanh(z4)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.base = BASE_DIM
        base = self.base

        self.conv1 = nn.Conv2d(3, base, (5, 5), padding=2, stride=2)
        self.bn1 = nn.BatchNorm2d(base, affine=True)
        self.conv2 = nn.Conv2d(base, base * 2, (5, 5), padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(base * 2, affine=True)
        self.conv3 = nn.Conv2d(base * 2, base * 4, (5, 5), padding=2, stride=2)
        self.bn3 = nn.BatchNorm2d(base * 4, affine=True)
        self.conv4 = nn.Conv2d(base * 4, base * 8, (5, 5), padding=2, stride=2)
        self.bn4 = nn.BatchNorm2d(base * 8, affine=True)
        self.collapse = nn.Linear((base * 8) * 4 * 4, 1)

        init_modules(self)
        self.clip()

    def forward(self, x):
        z1 = lrelu(self.bn1(self.conv1(x)))
        z2 = lrelu(self.bn2(self.conv2(z1)))
        z3 = lrelu(self.bn3(self.conv3(z2)))
        z4 = lrelu(self.bn4(self.conv4(z3)))
        return self.collapse(z4.view(-1, (self.base * 8) * 4 * 4))

    def clip(self):
        for m in self.modules():
            if not isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.data.clamp_(-CLIP, CLIP)


class WGANTrainingRunner(TrainingRunner):
    def __init__(self):
        super().__init__('wgan', create_dogs_dataset(),
                         Generator(), Discriminator(),
                         lambda n: torch.randn((n, NOISE_DIM)),
                         use_half=True)

        self.dsc_opt = optim.RMSprop(self.dsc.parameters(), lr=LEARNING_RATE, eps=1e-4)
        self.gen_opt = optim.RMSprop(self.gen.parameters(), lr=LEARNING_RATE, eps=1e-4)

    def run_epoch(self, it, context):
        while True:
            try:
                for _ in range(N_CRITIC):
                    X_real = next(it)
                    X_real = self.convert(X_real)
                    self.dsc.zero_grad()
                    self.gen.zero_grad()
                    X_real = self.convert(X_real)
                    noise = self.gen_noise(X_real.shape[0])
                    X_fake = self.gen(noise)
                    critic_f = self.dsc(X_fake)
                    critic_r = self.dsc(X_real)
                    critic_loss = torch.mean(critic_f) - torch.mean(critic_r)

                    critic_loss.backward()

                    critic_grad = grad_norm(self.dsc)

                    self.dsc_opt.step()
                    self.dsc.clip()

                self.dsc.zero_grad()
                self.gen.zero_grad()
                noise = self.gen_noise(BATCH_SIZE)
                gen_loss = -torch.mean(self.dsc(self.gen(noise)))
                gen_loss.backward()

                gen_grad = grad_norm(self.gen)

                self.gen_opt.step()

                context.add_scalar('loss/critic', critic_loss)
                context.add_scalar('loss/gen', gen_loss)

                context.add_scalar('grad/critic', critic_grad)
                context.add_scalar('grad/gen', gen_grad)

                context.inc_iter()
            except StopIteration:
                break


if __name__ == '__main__':
    runner = WGANTrainingRunner()
    runner.run(batch_size=BATCH_SIZE)