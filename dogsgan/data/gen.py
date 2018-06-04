import torch.utils.data as data
from torchvision import datasets, transforms


def generated_images_dataset(gen, size=10000):
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