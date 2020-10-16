import argparse

import torch

from model import Glow
from samplers import memory_mnist, memory_fashion
from utils import net_args, string_args

parser = net_args(argparse.ArgumentParser(description="Glow trainer"))
parser.add_argument("model_path", type=str, help="path to model weights")


def test(args, model):
    if args.dataset == "mnist":
        dataset_f = memory_mnist
    elif args.dataset == "fashion_mnist":
        dataset_f = memory_fashion
    args.delta = float(args.model_path.split(";")[-1].split("_")[0].split("#")[1])
    repr_args = string_args(args)
    f = open(f"./test/ll_per_point_{repr_args}_.txt", "w")
    train_loader, val_loader, train_val_loader, train_labels, val_labels = dataset_f(
        1, args.img_size, args.n_channels, return_y=True
    )
    with torch.no_grad():
        for ind, image in enumerate(train_loader):
            image = image.repeat(200, 1, 1, 1)
            image = image.to(device)
            image = image + torch.randn_like(image) * args.delta
    #         print(image)
            print(ind, end=",")
            log_p, log_det, _ = model(image)
            for i in range(log_p.shape[0]):
                print(ind, args.delta, log_p[i].item(), log_det[i].item(), train_labels[ind].item(), file=f)
            if ind >= 9999:
                break
    f.close()


if __name__ == "__main__":
    args = parser.parse_args()
    print(string_args(args))
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
