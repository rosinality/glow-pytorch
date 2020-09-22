import argparse

import torch

from model import Glow
from samplers import sample_data, memory_mnist
from utils import net_args, calc_loss

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = net_args(argparse.ArgumentParser(description='Glow trainer'))
parser.add_argument('model_path', type=str,
                    help='path to model weights')
parser.add_argument('path', metavar='PATH', type=str, help='Path to image directory')


def test(args, model):
    dataset = iter(memory_mnist(args.batch * 10, args.img_size, args.n_channels))
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
                loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins, args.n_channels)
                print(args.delta, log_p.item(), log_det.item(), cls, file=f)
    f.close()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    model_single = Glow(
        args.n_channels, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    model = model_single
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)

    test(args, model)
