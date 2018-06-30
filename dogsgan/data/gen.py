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

import torch.utils.data as data
from torchvision import datasets, transforms


def generated_images_dataset(gen, size=1000):
    class MyDataSet(data.Dataset):
        def __getitem__(self, index):
            noise = gen.gen_noise(1)
            pic = gen(noise).cpu()
            image = transforms.ToPILImage()(pic[0])
            image = transforms.Resize((299, 299))(image)
            tensor = transforms.ToTensor()(image)
            tensor = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(tensor)
            return tensor, 0

        def __len__(self):
            return size

    return MyDataSet()