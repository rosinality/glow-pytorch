import argparse
import logging
from datetime import datetime, timedelta
from math import log
from time import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm

from model import Glow, InvConv2d, InvConv2dLU

LOGGING_LEVEL = logging.DEBUG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGING_LEVEL = logging.DEBUG
parser = argparse.ArgumentParser(description="Glow trainer")
parser.add_argument("--batch", default=16, type=int, help="batch size")
parser.add_argument("--iter", default=200, type=int, help="maximum iterations")

parser.add_argument(
    "--n_flow", default=32, type=int, help="number of flows in each block"
)
parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
parser.add_argument(
    "--no_lu",
    action="store_true",
    help="use plain convolution instead of LU decomposed version",
)
parser.add_argument(
    "--affine", action="store_true", help="use affine coupling instead of additive"
)
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--img_size", default=64, type=int, help="image size")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=20, type=int, help="number of samples")
parser.add_argument("path", metavar="PATH", type=str, help="Path to image directory")


def sample_data(path, batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
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


def train(args, model, optimizer):
    logging.basicConfig(level=LOGGING_LEVEL)
    logger = logging.getLogger(train.__name__)

    dataset = iter(sample_data(args.path, args.batch, args.img_size))
    n_bins = 2.0 ** args.n_bits

    z_sample = []
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))
    exec_time_total = 0.0
    with tqdm(range(args.iter)) as pbar:
        for i in pbar:
            start_time = time()
            image, _ = next(dataset)
            image = image.to(device)

            image = image * 255

            if args.n_bits < 8:
                image = torch.floor(image / 2 ** (8 - args.n_bits))

            image = image / n_bins - 0.5

            if i == 0:
                with torch.no_grad():
                    log_p, logdet, _ = model.module(
                        image + torch.rand_like(image) / n_bins
                    )

                    continue

            else:
                log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins)

            logdet = logdet.mean()

            loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
            model.zero_grad()
            loss.backward()
            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            warmup_lr = args.lr
            optimizer.param_groups[0]["lr"] = warmup_lr
            optimizer.step()

            pbar.set_description(
                f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}"
            )

            if i % 100 == 0:
                with torch.no_grad():
                    utils.save_image(
                        model_single.reverse(z_sample).cpu().data,
                        f"sample/{str(i + 1).zfill(6)}.png",
                        normalize=True,
                        nrow=10,
                        range=(-0.5, 0.5),
                    )

            if i % 10000 == 0:
                torch.save(
                    model.state_dict(), f"checkpoint/model_{str(i + 1).zfill(6)}.pt"
                )
                torch.save(
                    optimizer.state_dict(), f"checkpoint/optim_{str(i + 1).zfill(6)}.pt"
                )
            end_time = time()
            exec_time_total += end_time - start_time
            avg_exec_time = float(exec_time_total) / (i + 1)
            logger.debug(f'\nFor iteration {i} out of {args.iter} , average execution time so far is '
                         f'{avg_exec_time} seconds\n')
            logger.debug(
                f'\nexpected finish time = {datetime.now() + timedelta(seconds=(args.iter - i) * avg_exec_time)}\n')



def play_w_model(model):
    logging.basicConfig(level=LOGGING_LEVEL)
    logger = logging.getLogger(play_w_model.__name__)
    logger.debug(f'model has {len(model.blocks)} blocks')
    for i, b in enumerate(model.blocks):
        logger.debug(f'block[{i}] has {len(b.flows)} flow')
        for j, f in enumerate(b.flows):
            if isinstance(f.invconv, InvConv2d):
                logger.debug(f'flow[{j}] is of type {str(InvConv2d)} with weight of size {f.invconv.weight.size()}')
            elif isinstance(f.invconv, InvConv2dLU):
                logger.debug(f'flow[{j}] is of type {str(InvConv2dLU)} with wp,wu,wl of size of sizes'
                             f'{f.invconv.w_p.size(), f.invconv.w_u.size(), f.invconv.w_l.size()}')
def dump_conv_W(model):
    logging.basicConfig(level=LOGGING_LEVEL)
    logger = logging.getLogger(dump_conv_W.__name__)
    logger.debug(f'model has {len(model.module.blocks)} blocks')
    for i, b in enumerate(model.module.blocks):
        logger.debug(f'block[{i}] has {len(b.flows)} flow')
        for j, f in enumerate(b.flows):
            if isinstance(f.invconv, InvConv2d):
                logger.debug(
                    f'flow[{j}] is of type {str(InvConv2d)} with weight of mean {torch.mean(f.invconv.weight)}')
            elif isinstance(f.invconv, InvConv2dLU):
                logger.debug(f'flow[{j}] is of type {str(InvConv2dLU)} with wp,wu,wl of ,means'
                             f'{torch.mean(f.invconv.w_p), torch.mean(f.invconv.w_u), torch.mean(f.invconv.w_l)}')


if __name__ == "__main__":
    logging.basicConfig(level=LOGGING_LEVEL)
    logger = logging.getLogger(__name__)
    args = parser.parse_args()
    logger.info(args)
    model_single = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=False
    )
    # play with model_single
    # play_w_model(model_single)
    # FIXME remove
    # logger.debug('Premature exit !!!')
    # sys.exit(-1)
    model = nn.DataParallel(model_single)
    # model = model_single
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    logger.debug(f'model-single instance is of type {type(model_single)}')
    logger.debug(f'model instance is of type {type(model)}')

    logger.debug(f'dump conv w before train')
    dump_conv_W(model)

    train(args, model, optimizer)
    # todo : model weights after training should be diff than before , given sufficient iter (sanity check)
    logger.debug(f'dump conv w after train')
    dump_conv_W(model)

    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    path = './model_single.dict'
    torch.save(model.module.state_dict(), path)
    model_module_loaded = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=False
    )
    model_module_loaded.load_state_dict(torch.load(path))
