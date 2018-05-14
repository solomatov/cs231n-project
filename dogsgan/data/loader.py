import torch

from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from dogsgan.data.common import preprocessed_dir


def create_loader(batch_size=128):

    trans = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Lambda(lambda img: (img - 128.0) / 128.0)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder(str(preprocessed_dir), transform=trans)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


if __name__ == '__main__':
    for i, (X, y) in enumerate(create_loader()):
        print(i)
