import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset, Dataset
from torchvision import transforms, datasets


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[index]

        if self.transform:
            x = self.transform(x)

        return x

    def __len__(self):
        return self.tensors.size(0)


def sample_data(path, batch_size, image_size, n_channels):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * n_channels, (1,) * n_channels),
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


def memory_mnist(batch_size, image_size, n_channels):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * n_channels, (1,) * n_channels),
        ]
    )
    data = torchvision.datasets.MNIST(
        "~/datasets/mnist/", train=True, download=True, transform=transform
    )

    train_data = CustomTensorDataset(data.data[:55000].clone(), transform=transform)
    val_data = CustomTensorDataset(data.data[55000:].clone(), transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader



def memory_fashion(batch_size, image_size, n_channels):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * n_channels, (1,) * n_channels),
        ]
    )
    data = torchvision.datasets.FashionMNIST(
        "~/datasets/fashion_mnist/", train=True, download=True, transform=transform
    )

    train_data = CustomTensorDataset(data.data[:55000].clone(), transform=transform)
    val_data = CustomTensorDataset(data.data[55000:].clone(), transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader
