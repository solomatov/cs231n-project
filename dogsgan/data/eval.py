import torch
import torch.nn.functional as F

from tqdm import tqdm

from torchvision.models.inception import inception_v3
from torchvision import transforms
from torch.utils.data import DataLoader


from dogsgan.data.dogs import create_dogs_dataset


def kl_divergence(p, q):
    return torch.sum(p * torch.log(p / q))


def inception_score(images):
    inception = inception_v3(pretrained=True)

    has_cuda = torch.cuda.device_count() > 0
    if has_cuda:
        inception = inception.cuda()

    inception.train(False)

    with torch.no_grad():
        scores = []
        for X, _ in tqdm(images):
            if has_cuda:
                X = torch.Tensor(X).cuda()

            scores.append(F.softmax(inception(X), 1))

        scores = torch.cat(scores, 0)

    sums = torch.sum(scores, 0)
    mdist = sums / torch.sum(sums)

    divs = []
    for i in range(0, scores.shape[0]):
        divs.append(kl_divergence(scores[i, :], mdist))

    print(torch.Tensor(divs).mean().exp())


if __name__ == '__main__':
    dogs_dataset = create_dogs_dataset(
        normalize=False,
        image_transforms=[transforms.Resize((299, 299))]
    )
    dataloader = DataLoader(dogs_dataset, batch_size=32)
    print(inception_score(dataloader))