import torch

from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from dogsgan.data.common import preprocessed_dir


def create_loader(batch_size=4):
    dataset = datasets.ImageFolder(str(preprocessed_dir), transform=transforms.ToTensor())
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


if __name__ == '__main__':
    for i, (X, y) in enumerate(create_loader()):
        print(i)
