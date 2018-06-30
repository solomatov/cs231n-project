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

        runner.train(epochs=100, batch_size=64)