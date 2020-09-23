import argparse
from argparse import Namespace

import torch
from torch import optim
from torchvision import utils
from tqdm import tqdm

from model import Glow
from samplers import memory_mnist, memory_fashion
from utils import net_args, calc_z_shapes, calc_loss, string_args

parser = net_args(argparse.ArgumentParser(description="Glow trainer"))


def train(args, model, optimizer):
    if args.dataset == "mnist":
        dataset_f = memory_mnist
    elif args.dataset == "fashion_mnist":
        dataset_f = memory_fashion
    dataset = iter(dataset_f(args.batch, args.img_size, args.n_channels))
    repr_args = string_args(args)
    n_bins = 2.0 ** args.n_bits

    z_sample = []
    z_shapes = calc_z_shapes(args.n_channels, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))

    losses = []
    with tqdm(range(args.iter)) as pbar:
        for i in pbar:
            image, _ = next(dataset)
            image = image.to(device)

            if i == 0:
                with torch.no_grad():
                    log_p, logdet, _ = model(
                        image + torch.randn_like(image) * args.delta
                    )

                    continue

            else:
                log_p, logdet, _ = model(image + torch.randn_like(image) * args.delta)

            logdet = logdet.mean()

            loss, log_p, log_det = calc_loss(
                log_p, logdet, args.img_size, n_bins, args.n_channels
            )
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            warmup_lr = args.lr
            optimizer.param_groups[0]["lr"] = warmup_lr
            optimizer.step()

            pbar.set_description(
                f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}"
            )

            if (i + 1) % 1000 == 0:
                with torch.no_grad():
                    utils.save_image(
                        model.reverse(z_sample).cpu().data,
                        f"sample/sample_{repr_args}_{str(i + 1).zfill(6)}.png",
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )
    torch.save(model.state_dict(), f"checkpoint/model_{repr_args}_.pt")

    f = open(f"ll/ll_{repr_args}_.txt", "w")
    for i in range(100):
        with torch.no_grad():
            image, _ = next(dataset)
            image = image.to(device)
            log_p, logdet, _ = model(image + torch.randn_like(image) * args.delta)
            logdet = logdet.mean()
            loss, log_p, log_det = calc_loss(
                log_p, logdet, args.img_size, n_bins, args.n_channels
            )
            print(args.delta, log_p.item(), log_det.item(), file=f)
    f.close()
    f = open(f"losses/losses_{repr_args}_txt", "w")
    print("\n".join(map(str, losses)), file=f)
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
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(args, model, optimizer)
