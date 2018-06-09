import argparse

from dogsgan.data.dogs import create_dogs_dataset
from dogsgan.training.optimizers import WGANGPOptimizer
from dogsgan.training.runner import TrainingRunner
from dogsgan.scripts.util import execute

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
    batch_size = args.batch_size
    load_from = args.load_from
    mode = args.mode

    runner = TrainingRunner('wgan_gp', create_dogs_dataset(),
                            dogs.Generator(noise_dim=noise_dim, base_dim=base_dim),
                            dogs.Discriminator(base_dim=base_dim, batch_norm=False), WGANGPOptimizer(lr=lr, l=l), args=args)

    execute(runner, load_from=load_from, mode=mode, batch_size=batch_size)