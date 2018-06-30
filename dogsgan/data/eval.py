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

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.inception import inception_v3
from tqdm import tqdm

from dogsgan.data.dogs import create_dogs_dataset


def kl_divergence(p, q):
    return torch.sum(p * torch.log(p / q))


def apply_classifier(classifier, images):
    has_cuda = torch.cuda.device_count() > 0
    with torch.no_grad():
        scores = []
        for X, _ in tqdm(DataLoader(images, batch_size=64), desc='Inception Score'):
            if has_cuda:
                X = torch.Tensor(X).cuda()
            scores.append(F.softmax(classifier(X), 1))
        scores = torch.cat(scores, 0)

    return scores


def compute_marginal(dists):
    sums = torch.sum(dists, 0)
    return sums / torch.sum(sums)


def inception_score(images):
    inception = inception_v3(pretrained=True)
    inception.train(False)

    has_cuda = torch.cuda.device_count() > 0
    if has_cuda:
        inception = inception.cuda()

    scores = apply_classifier(inception, images)
    mdist = compute_marginal(scores)

    divs = []
    for i in range(0, scores.shape[0]):
        divs.append(kl_divergence(scores[i, :], mdist))

    sd = torch.Tensor(divs).std().exp().item()
    return torch.Tensor(divs).mean().exp().item(), sd / (scores.shape[0] ** 0.5)


if __name__ == '__main__':
    dogs_dataset = create_dogs_dataset(
        normalize=False,
        image_transforms=[transforms.Resize((299, 299))]
    )
    print(inception_score(dogs_dataset))

    #14.634567260742188, 0.02150835660283012