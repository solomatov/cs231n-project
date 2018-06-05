from torch.utils.data import DataLoader
from dogsgan.data.dogs import create_dogs_dataset
from torchvision.utils import make_grid, save_image

if __name__ == '__main__':
    loader = DataLoader(create_dogs_dataset(), batch_size=104, shuffle=True)
    for X, _ in loader:
        save_image(make_grid(X, normalize=True), 'dogs.png')
        break