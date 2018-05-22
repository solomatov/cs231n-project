import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms

from dogsgan.infra.runner import TrainingRunner

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


class Generator(nn.Module):
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


class Critic(nn.Module):
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
        super().__init__('wgan-mnist', mnist)
        self.gen = Generator().to(self.device)
        self.critic = Critic().to(self.device)

        self.critic_opt = optim.RMSprop(self.critic.parameters(), lr=LEARNING_RATE)
        self.gen_opt = optim.RMSprop(self.gen.parameters(), lr=LEARNING_RATE)

        self.vis_params = torch.randn((104, NOISE_DIM)).to(self.device)

    def run_epoch(self, it, context):
        while True:
            try:
                for _ in range(N_CRITIC):
                    X_real, _ = next(it)
                    self.critic.zero_grad()
                    self.gen.zero_grad()
                    X_real = X_real.to(self.device)
                    X_fake = self.gen(torch.randn((X_real.shape[0], NOISE_DIM)).to(self.device))
                    critic_f = self.critic(X_fake)
                    critic_r = self.critic(X_real)
                    critic_loss = torch.mean(critic_f) - torch.mean(critic_r)
                    critic_loss.backward()
                    self.critic_opt.step()
                    self.critic.clip()

                self.critic.zero_grad()
                self.gen.zero_grad()
                noise = torch.randn((BATCH_SIZE, NOISE_DIM)).to(self.device)
                gen_loss = -torch.mean(self.critic(self.gen(noise)))
                gen_loss.backward()
                self.gen_opt.step()

                context.add_scalar('critic_loss', critic_loss)
                context.add_scalar('gen_loss', gen_loss)

                context.inc_iter()
            except StopIteration:
                break

    def sample_images(self):
        return self.gen(self.vis_params)

    def get_snapshot(self):
        return {
            'gen': self.gen,
            'critic': self.critic
        }


if __name__ == '__main__':
    runner = WGANTrainingRunner()
    runner.run(batch_size=BATCH_SIZE)

