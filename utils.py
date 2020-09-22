from math import log


def net_args(parser):
    parser.add_argument('--batch', default=16, type=int, help='batch size')
    parser.add_argument('--n_channels', default=1, type=int, help='number of image channels')
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
    parser.add_argument('--temp', default=0.7, type=float,
                        help='temperature of sampling')
    parser.add_argument('--n_sample', default=20, type=int, help='number of samples')
    parser.add_argument('--delta', default=0.01, type=float,
                        help='standard deviation of the de-quantizing noise')
    return parser


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins, n_dim):
    n_pixel = image_size * image_size * n_dim

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )