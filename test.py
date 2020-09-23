import argparse

import torch

from model import Glow
from samplers import memory_mnist, memory_fashion
from utils import net_args, calc_loss

parser = net_args(argparse.ArgumentParser(description="Glow trainer"))
parser.add_argument("model_path", type=str, help="path to model weights")


def test(args, model):
    if args.dataset == "mnist":
        dataset_f = memory_mnist
    elif args.dataset == "fashion_mnist":
        dataset_f = memory_fashion
    else:
        raise ValueError(f"Unknown dataset {args.dataset}!")
    dataset = iter(dataset_f(args.batch, args.img_size, args.n_channels))

    model.eval()
    n_bins = 2.0 ** args.n_bits
    f = open(f"./test/ll_{str(args.delta)}_.txt", "w")
    for i in range(100):
        with torch.no_grad():
            image_original, y = next(dataset)
            for cls in range(10):
                image = image_original[y == cls]
                print(image.shape)
                image = image.to(device)
                log_p, logdet, _ = model(image + torch.randn_like(image) * args.delta)
                logdet = logdet.mean()
                loss, log_p, log_det = calc_loss(
                    log_p, logdet, args.img_size, n_bins, args.n_channels
                )
                print(args.delta, log_p.item(), log_det.item(), cls, file=f)
    f.close()


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    device = args.device

    model_single = Glow(
        args.n_channels,
        args.n_flow,
        args.n_block,
        affine=args.affine,
        conv_lu=not args.no_lu,
    )
    model = model_single
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)

    test(args, model)
