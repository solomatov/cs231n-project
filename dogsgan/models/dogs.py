"""
   Copyright 2018 JetBrains, s.r.o
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dogsgan.training.runner import BaseGenerator
from dogsgan.models.util import init_weights


class Generator(BaseGenerator):
    def __init__(self, base_dim=128, noise_dim=1024, alpha=0.2, initializer=init_weights, affine=True):
        super().__init__()

        self.base_dim = base_dim
        self.noise_dim = noise_dim
        self.alpha = alpha

        self.noise_project = nn.Linear(noise_dim, 4 * 4 * base_dim * 8)
        self.bn0 = nn.BatchNorm1d(4 * 4 * base_dim * 8, affine=affine)
        self.conv1 = nn.ConvTranspose2d(base_dim * 8, base_dim * 4, (5, 5), stride=2, padding=2, output_padding=1)
        self.bn1 = nn.BatchNorm2d(base_dim * 4, affine=affine)
        self.conv2 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, (5, 5), stride=2, padding=2, output_padding=1)
        self.bn2 = nn.BatchNorm2d(base_dim * 2, affine=affine)
        self.conv3 = nn.ConvTranspose2d(base_dim * 2, base_dim, (5, 5), stride=2, padding=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(base_dim, affine=affine)
        self.conv4 = nn.ConvTranspose2d(base_dim, 3, (5, 5), stride=2, padding=2, output_padding=1)

        initializer(self)

    def forward(self, z):
        def lrelu(x):
            return F.leaky_relu(x, self.alpha)

        z0 = lrelu(self.bn0(self.noise_project(z)).view(-1, self.base_dim * 8, 4, 4))
        z1 = lrelu(self.bn1(self.conv1(z0)))
        z2 = lrelu(self.bn2(self.conv2(z1)))
        z3 = lrelu(self.bn3(self.conv3(z2)))
        z4 = self.conv4(z3)
        return F.tanh(z4)

    def gen_noise(self, n):
        return torch.randn((n, self.noise_dim), device=self.get_device())


class Discriminator(nn.Module):
    def __init__(self, base_dim=128, alpha=0.2, batch_norm=True, affine=True, initalizer=init_weights, clip_size=0.02):
        super().__init__()

        self.base_dim = base_dim
        self.alpha = alpha
        self.clip_size = clip_size

        self.conv1 = nn.Conv2d(3, base_dim, (5, 5), padding=2, stride=2)
        self.n1 = nn.BatchNorm2d(base_dim, affine=affine) if batch_norm else nn.LayerNorm([base_dim, 32, 32])
        self.conv2 = nn.Conv2d(base_dim, base_dim * 2, (5, 5), padding=2, stride=2)
        self.n2 = nn.BatchNorm2d(base_dim * 2, affine=affine) if batch_norm else nn.LayerNorm([base_dim * 2, 16, 16])
        self.conv3 = nn.Conv2d(base_dim * 2, base_dim * 4, (5, 5), padding=2, stride=2)
        self.n3 = nn.BatchNorm2d(base_dim * 4, affine=affine) if batch_norm else nn.LayerNorm([base_dim * 4, 8, 8])
        self.conv4 = nn.Conv2d(base_dim * 4, base_dim * 8, (5, 5), padding=2, stride=2)
        self.n4 = nn.BatchNorm2d(base_dim * 8, affine=affine) if batch_norm else nn.LayerNorm([base_dim * 8, 4, 4])
        self.collapse = nn.Linear((base_dim * 8) * 4 * 4, 1)

        initalizer(self)

    def forward(self, x):
        def lrelu(x):
            return F.leaky_relu(x, self.alpha)

        z1 = lrelu(self.n1(self.conv1(x)))
        z2 = lrelu(self.n2(self.conv2(z1)))
        z3 = lrelu(self.n3(self.conv3(z2)))
        z4 = lrelu(self.n4(self.conv4(z3)))
        z5 = self.collapse(z4.view(-1, (self.base_dim * 8) * 4 * 4))

        return z5

    def clip(self):
        for m in self.modules():
            if not isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.data.clamp_(-self.clip_size, self.clip_size)

