from dogsgan.data.dogs import create_dogs_dataset
from dogsgan.training.optimizers import WGANGPOptimizer
from dogsgan.training.runner import TrainingRunner

import dogsgan.models.dogs as dogs

if __name__ == '__main__':
    for l in [1.0, 2.0, 4.0, 8.0, 10.0, 16.0]:
        runner = TrainingRunner(
            f'wgan_gp-{l}', create_dogs_dataset(),
            dogs.Generator(base_dim=224),
            dogs.Discriminator(batch_norm=False, base_dim=224),
            WGANGPOptimizer(l=l))

        runner.run(epochs=100, batch_size=64)