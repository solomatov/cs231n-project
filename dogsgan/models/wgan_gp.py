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


class Generator(nn.Module):
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


class WGANTrainingRunner(TrainingRunner):
    def __init__(self):
        super().__init__('wgan-gp', create_dogs_dataset(), Generator(), Discriminator())

        self.dsc_opt = optim.Adam(self.dsc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))

    def run_epoch(self, it, context):
        while True:
            try:
                for _ in range(N_CRITIC):
                    X_real = next(it)

                    X_real = self.convert(X_real)
                    N = X_real.shape[0]

                    self.dsc.zero_grad()
                    self.gen.zero_grad()
                    X_real = self.convert(X_real)
                    noise = self.gen_noise(N)
                    X_fake = self.gen(noise)
                    critic_f = self.dsc(X_fake)
                    critic_r = self.dsc(X_real)


                    a = self.convert(torch.zeros([N, 1, 1, 1]).uniform_())

                    X_mix = X_real * a + X_fake * (1.0 - a)
                    gp_grad = torch.autograd.grad(outputs=self.dsc(X_mix), inputs=X_mix,
                                                  grad_outputs=self.convert(torch.ones([N, 1])),
                                                  retain_graph=True, create_graph=True)[0]

                    gp = torch.mean((gp_grad.norm(dim=1).norm(dim=1).norm(dim=1) - 1)**2)
                    critic_loss = torch.mean(critic_f) - torch.mean(critic_r) + LAMBDA * gp

                    critic_loss.backward()

                    critic_grad = grad_norm(self.dsc)

                    self.dsc_opt.step()

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
                context.add_scalar('grad/gp', gp)

                context.inc_iter()
            except StopIteration:
                break

    def gen_noise(self, n):
        return self.convert(torch.randn((n, NOISE_DIM)))

    def get_snapshot(self):
        return {
            'gen': self.gen,
            'critic': self.dsc
        }

if __name__ == '__main__':
    runner = WGANTrainingRunner()
    runner.run(batch_size=BATCH_SIZE)