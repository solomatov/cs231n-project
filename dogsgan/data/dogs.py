from torchvision import datasets, transforms
from dogsgan.data.common import preprocessed_dir


def create_dogs_dataset():
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return datasets.ImageFolder(str(preprocessed_dir), transform=trans)