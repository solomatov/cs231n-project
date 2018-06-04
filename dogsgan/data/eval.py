from torchvision.models.inception import inception_v3


if __name__ == '__main__':
    inception = inception_v3(pretrained=True)

    print()