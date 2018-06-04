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