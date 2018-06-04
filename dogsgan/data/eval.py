import torch
import torch.nn.functional as F

from tqdm import tqdm

from torchvision.models.inception import inception_v3
from torchvision import transforms
from torch.utils.data import DataLoader


from dogsgan.data.dogs import create_dogs_dataset


def kl_divergence(p, q):
    return torch.sum(p * torch.log(p / q))


if __name__ == '__main__':
    inception = inception_v3(pretrained=True)
    inception.train(False)

    dogs_dataset = create_dogs_dataset(
        normalize=False,
        image_transforms=[transforms.Resize((299, 299))]
    )
    dataloader = DataLoader(dogs_dataset, batch_size=16)

    with torch.no_grad():
        scores = []
        count = 0
        for X, _ in tqdm(dataloader):
            scores.append(F.softmax(inception(X), 1))
            count += 1
            if count == 20:
                break

        scores = torch.cat(scores, 0)

    sums = torch.sum(scores, 0)
    mdist = sums / torch.sum(sums)

    divs = []
    for i in range(0, scores.shape[0]):
        divs.append(kl_divergence(scores[i, :], mdist))

    print(torch.Tensor(divs).mean().exp())