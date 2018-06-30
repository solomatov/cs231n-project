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
            dogs.Discriminator(base_dim=base_dim),
            VanillaGANOptimizer(dsc_lr=lr, gen_lr=lr))

        runner.train(epochs=100)