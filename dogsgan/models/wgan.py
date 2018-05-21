import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from dogsgan.infra.runner import TrainingRunner


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


class Critic(nn.Module):
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

        self.clip()

    def forward(self, x):
        z1 = lrelu(self.bn1(self.conv1(x)))
        z2 = lrelu(self.bn2(self.conv2(z1)))
        z3 = lrelu(self.bn3(self.conv3(z2)))
        z4 = lrelu(self.bn4(self.conv4(z3)))
        return self.collapse(z4.view(-1, (self.base * 8) * 4 * 4))

    def clip(self):
        for p in self.parameters():
            p.data.clamp_(-CLIP, CLIP)


class WGANTrainingRunner(TrainingRunner):
    def __init__(self):
        super().__init__('wgan')
        self.gen = Generator().to(self.device)
        self.critic = Critic().to(self.device)

        self.critic_opt = optim.RMSprop(self.critic.parameters(), lr=LEARNING_RATE)
        self.gen_opt = optim.RMSprop(self.gen.parameters(), lr=LEARNING_RATE)

        self.vis_params = torch.randn((104, NOISE_DIM)).to(self.device)

    def run_epoch(self, it):
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