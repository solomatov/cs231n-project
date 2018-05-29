import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms

from dogsgan.training.runner import TrainingRunner, BaseGenerator

ALPHA = 0.2
BATCH_SIZE = 64
N_CRITIC = 5
CLIP = 0.01
LEARNING_RATE = 0.00005
NOISE_DIM = 100
BASE_DIM = 64
WEIGHT_STD = 0.02


def lrelu(x):
    return F.leaky_relu(x, ALPHA)


class Generator(BaseGenerator):
    def __init__(self):
        super().__init__()

        self.base = BASE_DIM
        base = self.base

        self.l1 = nn.Linear(NOISE_DIM, 1024)
        self.bn1 = nn.BatchNorm1d(1024, affine=True)
        self.l2 = nn.Linear(1024, 7 * 7 * base * 2)
        self.bn2 = nn.BatchNorm1d(7 * 7 * base * 2)
        self.convt1 = nn.ConvTranspose2d(base * 2, base, (4, 4), stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(base)
        self.convt2 = nn.ConvTranspose2d(base, 1, (4, 4), stride=2, padding=1)

    def forward(self, z):
        z1 = lrelu(self.bn1(self.l1(z)))
        z2 = lrelu(self.bn2(self.l2(z1)))
        z3 = z2.view(-1, self.base * 2, 7, 7)
        z4 = lrelu(self.bn3(self.convt1(z3)))
        return F.tanh(self.convt2(z4))

    def gen_noise(self, n):
        return torch.randn((n, NOISE_DIM), device=self.get_device())


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.base = BASE_DIM
        base = self.base

        self.conv1 = nn.Conv2d(1, base, (5, 5))
        self.conv2 = nn.Conv2d(base, base * 2, (5, 5))
        self.l1 = nn.Linear(4 * 4 * base * 2, 1024)
        self.l2 = nn.Linear(1024, 1)

        self.clip()

    def forward(self, x):

        N = x.shape[0]

        z1 = lrelu(self.conv1(x))
        z2 = F.max_pool2d(z1, (2, 2), stride=2)
        z3 = lrelu(self.conv2(z2))
        z4 = F.max_pool2d(z3, (2, 2), stride=2)
        z5 = z4.view(N, -1)
        z6 = lrelu(self.l1(z5))
        score = self.l2(z6)
        return score

    def clip(self):
        for m in self.modules():
            if not isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.data.clamp_(-CLIP, CLIP)


mnist = datasets.MNIST(root='data/mnist', download=True, transform=transforms.ToTensor())


class WGANTrainingRunner(TrainingRunner):
    def __init__(self):
        super().__init__('wgan-mnist', mnist,
                         Generator(), Discriminator(), None)

        self.dsc_opt = optim.RMSprop(self.dsc.parameters(), lr=LEARNING_RATE)
        self.gen_opt = optim.RMSprop(self.gen.parameters(), lr=LEARNING_RATE)

    def run_epoch(self, it, context):
        while True:
            try:
                for _ in range(N_CRITIC):
                    X_real = next(it)
                    self.dsc.zero_grad()
                    self.gen.zero_grad()
                    X_real = X_real.to(self.device)
                    X_fake = self.gen(self.gen.gen_noise(X_real.shape[0]))
                    critic_f = self.dsc(X_fake)
                    critic_r = self.dsc(X_real)
                    critic_loss = torch.mean(critic_f) - torch.mean(critic_r)
                    critic_loss.backward()
                    self.dsc_opt.step()
                    self.dsc.clip()

                self.dsc.zero_grad()
                self.gen.zero_grad()
                noise = self.gen.gen_noise(BATCH_SIZE)
                gen_loss = -torch.mean(self.dsc(self.gen(noise)))
                gen_loss.backward()
                self.gen_opt.step()

                context.add_scalar('critic_loss', critic_loss)
                context.add_scalar('gen_loss', gen_loss)

                context.inc_iter()
            except StopIteration:
                break


if __name__ == '__main__':
    runner = WGANTrainingRunner()
    runner.run(batch_size=BATCH_SIZE)

