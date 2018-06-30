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

import argparse

from dogsgan.data.dogs import create_dogs_dataset
from dogsgan.training.optimizers import WGANGPOptimizer
from dogsgan.training.runner import TrainingRunner

import dogsgan.models.dogs as dogs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run dcgan training')
    parser.add_argument('--lr', type=float, default=1e-4 ,help='learning rate')
    parser.add_argument('--l', type=float, default=10.0, help='gradient penalty coefficent')
    parser.add_argument('--noise_dim', type=int, default=1024, help='noise dimension')
    parser.add_argument('--base_dim', type=int, default=128, help='base dimension')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--load_from', type=str, default=None, help='directory to load from')
    parser.add_argument('--mode', type=str, default='train', help='mode train or evaluate')

    args = parser.parse_args()
    lr = args.lr
    l = args.l
    noise_dim = args.noise_dim
    base_dim = args.base_dim

    runner = TrainingRunner('wgan_gp', create_dogs_dataset(),
                            dogs.Generator(noise_dim=noise_dim, base_dim=base_dim),
                            dogs.Discriminator(base_dim=base_dim, batch_norm=False),
                            WGANGPOptimizer(lr=lr, l=l), args=args)

    runner.execute(args)