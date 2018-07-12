from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torch.distributions.multivariate_normal import MultivariateNormal

from model import Glow

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32
n_flow = 16
image_size = 64
n_block = 4
n_sample = 20
affine = True
n_iter = 500000
celeba_path = 'celeba'


transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
    ]
)


def sample_data():
    dataset = datasets.ImageFolder(celeba_path, transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=6)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=6
            )
            loader = iter(loader)
            yield next(loader)


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_log_p(z_list):
    normalize = -0.5 * log(2 * pi)
    log_p = 0
    for z in z_list:
        dist = -z ** 2 / 2
        p = normalize + dist
        p = p.view(batch_size, -1)
        log_p = log_p + p.sum(1)

    return log_p


def calc_loss(log_p, logdet):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -log(256) * n_pixel
    loss = loss + log_p + logdet

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


model_single = Glow(3, n_flow, n_block, affine=affine)
# model = nn.DataParallel(model_single)
model = model_single
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train():
    dataset = iter(sample_data())

    z_sample = []
    z_shapes = calc_z_shapes(3, image_size, n_flow, n_block)
    for z in z_shapes:
        z_new = torch.randn(n_sample, *z)
        z_sample.append(z_new.to(device))

    with tqdm(range(n_iter)) as pbar:
        for i in pbar:
            image, _ = next(dataset)
            image = image.to(device)
            log_p, logdet = model(image + torch.rand_like(image) / 256)
            loss, log_p, log_det = calc_loss(log_p, logdet)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(
                f'Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}'
            )

            if i % 100 == 0:
                with torch.no_grad():
                    utils.save_image(
                        model_single.reverse(z_sample).cpu().data,
                        f'sample/{str(i + 1).zfill(6)}.png',
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )

            if i % 10000 == 0:
                torch.save(model.state_dict(), f'checkpoint/model_{str(i + 1).zfill(6)}.pt')
                torch.save(optimizer.state_dict(), f'checkpoint/optim_{str(i + 1).zfill(6)}.pt')


if __name__ == '__main__':
    train()
