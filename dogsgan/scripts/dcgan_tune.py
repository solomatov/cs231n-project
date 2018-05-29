from dogsgan.data.dogs import create_dogs_dataset
from dogsgan.training.optimizers import VanillaGANOptimizer
from dogsgan.training.runner import TrainingRunner

import dogsgan.models.dogs as dogs

if __name__ == '__main__':
    lr = 2e-4
    noise_dim = 1024

    for base_dim in [16, 32, 64, 96, 128, 160, 196, 256, 320]:
        opt = VanillaGANOptimizer()
        runner = TrainingRunner(
            f'dcgan-{base_dim}', create_dogs_dataset(),
            dogs.Generator(base_dim=base_dim, noise_dim=noise_dim),
            dogs.Discriminator(base_dim=base_dim, sigmoid=True),
            VanillaGANOptimizer(dsc_lr=lr, gen_lr=lr))

        runner.run(epochs=100)