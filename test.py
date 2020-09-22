import argparse
from math import log

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import Glow

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Glow mnist test per class')
parser.add_argument('--batch', default=16, type=int, help='batch size')
parser.add_argument('--iter', default=200000, type=int, help='maximum iterations')
parser.add_argument(
    '--n_flow', default=32, type=int, help='number of flows in each block'
)
parser.add_argument('--n_block', default=4, type=int, help='number of blocks')
parser.add_argument(
    '--no_lu',
    action='store_true',
    help='use plain convolution instead of LU decomposed version',
)
parser.add_argument(
    '--affine', action='store_true', help='use affine coupling instead of additive'
)
parser.add_argument('--n_bits', default=5, type=int, help='number of bits')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--img_size', default=64, type=int, help='image size')
parser.add_argument('--temp', default=0.7, type=float, help='temperature of sampling')
parser.add_argument('--n_sample', default=20, type=int, help='number of samples')
parser.add_argument('--delta', default=0.01, type=float,
                    help='standard deviation of the de-quantizing noise')
parser.add_argument('model_path', type=str,
                    help='path to model weights')
parser.add_argument('path', metavar='PATH', type=str, help='Path to image directory')


def sample_data(path, batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
        ]
    )

    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
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


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def test(args, model):
    dataset = iter(sample_data(args.path, args.batch * 10, args.img_size))
    model.eval()
    n_bins = 2. ** args.n_bits
    f = open(f'./test/ll_{str(args.delta)}_.txt', 'w')
    for i in range(100):
        with torch.no_grad():
            image_original, y = next(dataset)
            for cls in range(10):
                image = image_original[y == cls]
                print(image.shape)
                image = image.to(device)
                log_p, logdet, _ = model(image + torch.randn_like(image) * args.delta)
                logdet = logdet.mean()
                loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
                print(args.delta, log_p.item(), log_det.item(), cls, file=f)
    f.close()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    model_single = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    model = model_single
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)

    test(args, model)
