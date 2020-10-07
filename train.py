import argparse

import numpy as np
import torch
from torch import optim
from torchvision import utils
from tqdm import tqdm

from model import Glow
from samplers import memory_mnist, memory_fashion, point_2d
from utils import net_args, calc_z_shapes, calc_loss, string_args

parser = net_args(argparse.ArgumentParser(description="Glow trainer"))


def train(args, model, optimizer):
    if args.dataset == "mnist":
        dataset_f = memory_mnist
    elif args.dataset == "fashion_mnist":
        dataset_f = memory_fashion
    elif args.dataset == "point_2d":
        dataset_f = point_2d

    repr_args = string_args(args)
    n_bins = 2.0 ** args.n_bits

    z_sample = []
    z_shapes = calc_z_shapes(args.n_channels, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))

    epoch_losses = []
    min_loss = 1e12
    f_loss = open(f"losses/losses_{repr_args}_.txt", "w", buffering=1)
    with tqdm(range(args.epochs)) as pbar:
        for i in pbar:
            args.delta = 1000
            train_loader, val_loader = dataset_f(
                args.batch, args.img_size, args.n_channels
            )
            for image in train_loader:
                optimizer.zero_grad()
                image = image.to(device)
                log_p, logdet, _ = model(image + torch.randn_like(image) * args.delta)
                logdet = logdet.mean()
                loss, log_p, log_det = calc_loss(
                    log_p, logdet, args.img_size, n_bins, args.n_channels
                )
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                utils.save_image(
                    model.reverse(z_sample).cpu().data,
                    f"sample/sample_{repr_args}_{str(i + 1).zfill(6)}.png",
                    normalize=True,
                    nrow=10,
                    range=(-0.5, 0.5),
                )
            losses = []
            logdets = []
            logps = []
            for image in val_loader:
                image = image.to(device)
                log_p, logdet, _ = model(image + torch.randn_like(image) * args.delta)
                logdet = logdet.mean()
                loss, log_p, log_det = calc_loss(
                    log_p, logdet, args.img_size, n_bins, args.n_channels
                )
                losses.append(loss.item())
                logdets.append(log_det.item())
                logps.append(log_p.item())
            pbar.set_description(
                f"Loss: {np.mean(losses):.5f}; logP: {np.mean(logps):.5f}; logdet: {np.mean(logdets):.5f}"
            )
            current_loss = np.mean(losses)
            print(current_loss, file=f_loss)
            epoch_losses.append(current_loss)
            if current_loss <= min_loss:
                min_loss = current_loss
                torch.save(model.state_dict(), f"checkpoint/model_{repr_args}_.pt")
            if len(epoch_losses) >= 10 and min(epoch_losses[-10:]) > min_loss:
                break

    f_ll = open(f"ll/ll_{repr_args}_.txt", "w")
    _, val_loader = dataset_f(args.batch, args.img_size, args.n_channels)
    for image in val_loader:
        image = image.to(device)
        log_p, logdet, _ = model(image + torch.randn_like(image) * args.delta)
        logdet = logdet.mean()
        loss, log_p, log_det = calc_loss(
            log_p, logdet, args.img_size, n_bins, args.n_channels
        )
        print(args.delta, log_p.item(), log_det.item(), file=f_ll)
    f_ll.close()
    f_loss.close()


if __name__ == "__main__":
    args = parser.parse_args()
    print(string_args(args))
    device = args.device
    model = Glow(
        args.n_channels,
        args.n_flow,
        args.n_block,
        affine=args.affine,
        conv_lu=not args.no_lu,
    )
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train(args, model, optimizer)
