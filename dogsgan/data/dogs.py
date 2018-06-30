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

from torchvision import datasets, transforms
from dogsgan.data.common import preprocessed_dir


def create_dogs_dataset(normalize=True, image_transforms=None, additional_transforms=None):
    trans_list = []

    if image_transforms is not None:
        trans_list.extend(image_transforms)

    trans_list.append(transforms.ToTensor())

    if normalize:
        trans_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    if additional_transforms is not None:
        trans_list.extend(additional_transforms)

    return datasets.ImageFolder(str(preprocessed_dir), transform=transforms.Compose(trans_list))