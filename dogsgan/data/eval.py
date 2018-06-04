from torchvision.models.inception import inception_v3
from torchvision import transforms
from torch.utils.data import DataLoader


from dogsgan.data.dogs import create_dogs_dataset

if __name__ == '__main__':
    inception = inception_v3(pretrained=True)
    inception.train(False)

    dogs_dataset = create_dogs_dataset(
        normalize=False,
        image_transforms=[transforms.Resize((299, 299))]
    )
    dataloader = DataLoader(dogs_dataset, batch_size=16)

    for X, _ in dataloader:
        out = inception(X)
        print(out.shape)