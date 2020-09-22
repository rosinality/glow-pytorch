import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from train import N_DIM


def sample_data(path, batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * N_DIM, (1,) * N_DIM),
        ]
    )

    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=8)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)


def memory_mnist(batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * N_DIM, (1,) * N_DIM),
        ]
    )
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('~/users/pt/files/', train=True, download=True,
                                   transform=transform
                                   ),
        batch_size=batch_size, shuffle=True)

    loader = iter(train_loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST('~/users/pt/files/', train=True,
                                           download=True,
                                           transform=transform
                                           ),
                batch_size=batch_size, shuffle=True)
            loader = iter(train_loader)
            yield next(loader)