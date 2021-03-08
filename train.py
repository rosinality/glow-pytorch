import argparse

import numpy as np
import torch
from torch import optim
from torchvision import utils
from tqdm import tqdm

from model import Glow
from samplers import memory_mnist, memory_fashion, celeba
from utils import (
    net_args,
    calc_z_shapes,
    calc_loss,
    string_args,
)

parser = net_args(argparse.ArgumentParser(description="Glow trainer"))


def train(args, model, optimizer):
    if args.dataset == "mnist":
        dataset_f = memory_mnist
    elif args.dataset == "fashion_mnist":
        dataset_f = memory_fashion
    elif args.dataset == "celeba":
        dataset_f = celeba
    else:
        raise ValueError("Unknown dataset:", args.dataset)

    repr_args = string_args(args)
    n_bins = 2.0 ** args.n_bits

    z_sample = []
    z_shapes = calc_z_shapes(args.n_channels, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))

    epoch_losses = []
    f_train_loss = open(f"losses/losses_train_{repr_args}_.txt", "a", buffering=1)
    f_test_loss = open(f"losses/losses_test_{repr_args}_.txt", "a", buffering=1)

    last_model_path = f"checkpoint/model_{repr_args}_last_.pt"
    try:
        model.load_state_dict(torch.load(last_model_path))
        model.eval()
        f_epoch = open(f"checkpoint/last_epoch_{repr_args}.txt", "r", buffering=1)
        epoch_n = int(f_epoch.readline().strip())
        f_epoch.close()
    except FileNotFoundError:
        print("Training the model from scratch.")
        epoch_n = 0

    with tqdm(range(epoch_n, args.epochs + epoch_n)) as pbar:
        for i in pbar:
            repr_args = string_args(args)
            train_loader, val_loader, train_val_loader = dataset_f(
                args.batch, args.img_size, args.n_channels
            )
            train_losses = []
            for image in train_loader:
                if isinstance(image, list):
                    image = image[0]
                optimizer.zero_grad()
                image = image.to(device)
                noisy_image = image
                if args.tr_dq:
                    noisy_image += torch.rand_like(image) / n_bins
                noisy_image += torch.randn_like(image) * args.delta
                log_p, logdet, _ = model(noisy_image)

                logdet = logdet.mean()
                loss, log_p, log_det = calc_loss(
                    log_p, logdet, args.img_size, n_bins, args.n_channels
                )
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            current_train_loss = np.mean(train_losses)
            print(f"{current_train_loss},{args.delta},{i + 1}", file=f_train_loss)
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
                    if isinstance(image, list):
                        image = image[0]
                    image = image.to(device)
                    log_p, logdet, _ = model(image)
                    logdet = logdet.mean()
                    loss, log_p, log_det = calc_loss(
                        log_p, logdet, args.img_size, n_bins, args.n_channels
                    )
                    losses.append(loss.item())
                    logdets.append(log_det.item())
                    logps.append(log_p.item())
                pbar.set_description(
                    f"Loss: {np.mean(losses):.5f}; logP: {np.mean(logps):.5f}; logdet: {np.mean(logdets):.5f}; delta: {args.delta:.5f}"
                )
                current_loss = np.mean(losses)
                print(f"{current_loss},{args.delta},{i + 1}", file=f_test_loss)
                epoch_losses.append(current_loss)
                if (i + 1) % 5 == 0:
                    torch.save(
                        model.state_dict(), f"checkpoint/model_{repr_args}_{i + 1}_.pt"
                    )
                torch.save(model.state_dict(), last_model_path)
                f_epoch = open(
                    f"checkpoint/last_epoch_{repr_args}.txt", "w", buffering=1
                )
                f_epoch.write(str(i + 1))
                f_epoch.close()

                f_ll = open(f"ll/ll_{repr_args}_{i + 1}.txt", "w")
                train_loader, val_loader, train_val_loader = dataset_f(
                    args.batch, args.img_size, args.n_channels
                )
                train_val_loader = iter(train_val_loader)
                for image_val in val_loader:
                    image = image_val
                    if isinstance(image, list):
                        image = image[0]
                    image = image.to(device)
                    log_p_val, logdet_val, _ = model(image)

                    image = next(train_val_loader)
                    if isinstance(image, list):
                        image = image[0]
                    image = image.to(device)
                    log_p_train_val, logdet_train_val, _ = model(image)

                    for (
                        lpv,
                        ldv,
                        lptv,
                        ldtv,
                    ) in zip(log_p_val, logdet_val, log_p_train_val, logdet_train_val):
                        print(
                            args.delta,
                            lpv.item(),
                            ldv.item(),
                            lptv.item(),
                            ldtv.item(),
                            file=f_ll,
                        )
                f_ll.close()
    f_train_loss.close()
    f_test_loss.close()


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
