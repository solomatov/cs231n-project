import argparse
from pathlib import Path

from dogsgan.data.dogs import create_dogs_dataset
from dogsgan.training.optimizers import VanillaGANOptimizer
from dogsgan.training.runner import TrainingRunner

import dogsgan.models.dogs as dogs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run dcgan training')
    parser.add_argument('--lr', type=float, default=2e-4 ,help='learning rate')
    parser.add_argument('--noise_dim', type=int, default=1024, help='noise dimension')
    parser.add_argument('--base_dim', type=int, default=128, help='base dimension')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--load_from', type=str, default=None, help='directory to load from')
    parser.add_argument('--loss', type=str, default='ls', help='loss function: bce or ls')

    args = parser.parse_args()
    lr = args.lr
    noise_dim = args.noise_dim
    base_dim = args.base_dim
    batch_size = args.batch_size
    load_from = args.load_from
    loss = args.loss

    opt = VanillaGANOptimizer()
    runner = TrainingRunner(
        'dcgan', create_dogs_dataset(),
        dogs.Generator(base_dim=base_dim, noise_dim=noise_dim),
        dogs.Discriminator(base_dim=base_dim),
        VanillaGANOptimizer(dsc_lr=lr, gen_lr=lr, loss=loss), args=args)

    if load_from is not None:
        print(f'Loading from {load_from}')
        runner.load_snapshot(Path(load_from))

    runner.run(batch_size=batch_size)