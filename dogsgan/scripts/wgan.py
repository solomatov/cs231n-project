import argparse

from dogsgan.data.dogs import create_dogs_dataset
from dogsgan.training.optimizers import WGANOptimizer
from dogsgan.training.runner import TrainingRunner

import dogsgan.models.dogs as dogs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run dcgan training')
    parser.add_argument('--lr', type=float, default=1e-4 ,help='learning rate')
    parser.add_argument('--clip', type=int, default=0.02, help='weight clip threshold')
    parser.add_argument('--noise_dim', type=int, default=1024, help='noise dimension')
    parser.add_argument('--base_dim', type=int, default=128, help='base dimension')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--load_from', type=str, default=None, help='directory to load from')

    args = parser.parse_args()
    lr = args.lr
    noise_dim = args.noise_dim
    base_dim = args.base_dim
    batch_size = args.batch_size
    load_from = args.load_from
    clip = args.clip

    runner = TrainingRunner('wgan', create_dogs_dataset(),
                            dogs.Generator(noise_dim=noise_dim, base_dim=base_dim, affine=False),
                            dogs.Discriminator(base_dim=base_dim, affine=False, clips_size=clip), WGANOptimizer(lr=lr), args=args)

    if load_from is not None:
        runner.load_snapshot(load_from)

    runner.run(batch_size=batch_size)