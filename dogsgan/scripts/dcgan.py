import argparse

from dogsgan.data.dogs import create_dogs_dataset
from dogsgan.training.optimizers import VanillayGANOptimizer
from dogsgan.training.runner import TrainingRunner

import dogsgan.models.dogs as dogs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run dcgan training')
    parser.add_argument('--lr', type=float, default=2e-4 ,help='learning rate')
    parser.add_argument('--noise_dim', type=int, default=1024, help='noise dimension')
    parser.add_argument('--base_dim', type=int, default=128, help='base dimension')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')

    args = parser.parse_args()
    lr = args.lr
    noise_dim = args.noise_dim
    base_dim = args.base_dim
    batch_size = args.batch_size

    opt = VanillayGANOptimizer()
    runner = TrainingRunner(
        'dcgan', create_dogs_dataset(),
        dogs.Generator(base_dim=base_dim, noise_dim=noise_dim),
        dogs.Discriminator(base_dim=base_dim, sigmoid=True),
        VanillayGANOptimizer(dsc_lr=lr, gen_lr=lr), args=args)

    runner.run(batch_size=batch_size)