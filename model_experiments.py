import matplotlib.pyplot as plt
import torch
import torchvision.utils
from PIL import Image
from torch import nn
from torchvision import transforms
from sklearn.metrics import mean_squared_error
from model import Glow
from train import calc_z_shapes
from utils import tensor_stat


def generate_img_tensor(img, image_size, n_bits):
    n_bins = 2 ** n_bits
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    img_tensor = transform(img).to(device).unsqueeze(0)
    img_tensor = img_tensor * 255
    if n_bits < 8:
        img_tensor = torch.floor(img_tensor / 2 ** (8 - n_bits))
    img_tensor = img_tensor / n_bins - 0.5
    return img_tensor


def encode(img_tensor, model, temp):
    _, _, z_outs = model(img_tensor)
    for i, z_out_i in enumerate(z_outs):
        # TODO investigate why we need to normalize z to make generated images looking good,
        # TODO By design, generated z should be normal , or ????
        norm_transform = transforms.Compose([transforms.Normalize(mean=torch.mean(z_out_i), std=torch.std(z_out_i))])
        z_outs[i] = norm_transform(z_out_i)
        z_outs[i] = z_outs[i] * temp
    return z_outs


def full_rank_encode_decode_experiment_2(model, img_filepath, nbits, img_size, temp, outfilename):
    img = Image.open(img_filepath)
    input_img_tensor = generate_img_tensor(img, img_size, nbits)
    z_outs = encode(input_img_tensor, model, temp)
    # decode
    generated_img_tensor = model.module.reverse(z_outs).cpu().data
    torchvision.utils.save_image(tensor=generated_img_tensor, fp=outfilename, normalize=True, range=(-0.5, 0.5))
    y = input_img_tensor.flatten().detach().numpy()
    y_hat = generated_img_tensor.flatten().detach().numpy()
    mse_ = mean_squared_error(y_true=y,y_pred=y_hat)
    print(mse_)


# FIXME experimental method, delete later !!!
def full_rank_encode_decode_draft(model, imgfile):
    path = 'celeba1/class1/000010.jpg'
    n_bits = 5
    n_bins = 2 ** 5
    image_size = 64
    img = Image.open(path)

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    img_tensor = transform(img).to(device).unsqueeze(0)
    torchvision.utils.save_image(img_tensor, fp='./startimage.png', normalize=True, range=(0, 1))
    stat_1 = tensor_stat(img_tensor)

    img_tensor = img_tensor * 255
    stat_2 = tensor_stat(img_tensor)
    if n_bits < 8:
        img_tensor = torch.floor(img_tensor / 2 ** (8 - n_bits))

    img_tensor = img_tensor / n_bins - 0.5

    stat_3 = tensor_stat(img_tensor)

    log_p_sum, logdet, z_outs = model(img_tensor)

    for i, z_out_i in enumerate(z_outs):
        transform2 = transforms.Compose([transforms.Normalize(mean=torch.mean(z_out_i), std=torch.std(z_out_i))])
        z_outs[i] = transform2(z_out_i)
        z_outs[i] = z_outs[i] * 0.7
    z_stat_1 = [tensor_stat(z_) for z_ in z_outs]
    for z_ in z_outs:
        plt.hist(torch.flatten(z_).detach().numpy())
        plt.show()
    gen_sample = model.module.reverse(z_outs).cpu().data

    stat_4 = tensor_stat(gen_sample)

    torchvision.utils.save_image(tensor=gen_sample, fp='./gen2.png', normalize=True, range=(-0.5, 0.5))
    img_size = 64
    n_flow = 32
    n_block = 4
    z_shapes = calc_z_shapes(3, img_size, n_flow, n_block)
    n_sample = 1
    z_sample = []
    for z in z_shapes:
        z_new = torch.randn(n_sample, *z) * temp
        z_sample.append(z_new.to(device))
    gen_sample2 = model.module.reverse(z_sample).cpu().data

    stat_5 = tensor_stat(gen_sample2)
    z_stat_2 = [tensor_stat(z_) for z_ in z_sample]
    for z_ in z_sample:
        plt.hist(torch.flatten(z_).detach().numpy())
        plt.show()
    torchvision.utils.save_image(tensor=gen_sample2, fp='./gen22.png', normalize=True, range=(-0.5, 0.5))


def rank_norm_experiment1(model):
    rank_norm_dict = dict()
    for f in model.module.blocks[3].flows:
        w = f.invconv.weight.squeeze()
        u, s, vh = torch.linalg.svd(w)
        for r in range(1, s.shape[0] + 1):
            m1 = torch.matmul(u[:, :r], torch.diag(s[:r]))
            w_ = torch.matmul(m1, vh[:r, :])
            rel_norm = torch.norm(w - w_) / torch.norm(w)
            if r in rank_norm_dict.keys():
                rel_norm_old, cnt = rank_norm_dict[r]
                rank_norm_dict[r] = rel_norm_old + rel_norm.item(), cnt + 1
            else:
                rank_norm_dict[r] = rel_norm.item(), 1
    x = []
    y = []
    for k in rank_norm_dict.keys():
        x.append(k)
        y.append(rank_norm_dict[k][0] / rank_norm_dict[k][1])
    plt.plot(x, y)
    plt.title("Truncation Rank vs rel-norm loss")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


if __name__ == '__main__':
    n_sample = 20
    img_size = 64
    n_flow = 32
    n_block = 4
    temp = 0.7
    n_bits = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    affine = False
    conv_lu = False
    model_single = Glow(3, n_flow, n_block, affine=affine, conv_lu=False)
    model = nn.DataParallel(model_single)

    model.load_state_dict(torch.load('checkpoint_2021-11-07T14:25:53.737008/model_006000.pt'))
    # Start experiments
    file_prefix = '000016'
    full_rank_encode_decode_experiment_2(model=model, img_filepath=f'celeba1/class1/{file_prefix}.jpg', nbits=n_bits,
                                         img_size=img_size,
                                         temp=temp, outfilename=f'gen{file_prefix}.png')

    # Draft code
    # # print(model.module.blocks[3].flows[0].invconv.weight.size())
    # # print(type(model.module.blocks[3].flows[0].invconv.weight[:, :, 0, 0].detach().cpu().numpy().shape))
    # # w_ = model.module.blocks[3].flows[0].invconv.weight[:, :, 0, 0].detach().cpu().numpy()
    # z_sample = []
    # z_shapes = calc_z_shapes(3, img_size, n_flow, n_block)
    # print('generating sample image')
    # gen_sample_img_path = './gen_sample9.png'
    # for z in z_shapes:
    #     z_new = torch.randn(n_sample, *z) * temp
    #     z_sample.append(z_new.to(device))
    # gen_sample = model.module.reverse(z_sample).cpu().data
    # torchvision.utils.save_image(tensor=gen_sample, fp=gen_sample_img_path, normalize=True,
    #                              nrow=10,
    #                              range=(-0.5, 0.5))

    # # quick test of matrix_rank function with SVD
    # eps_float32 = sys.float_info.epsilon
    # # TODO how to set tol
    # r = np.linalg.matrix_rank(M=w_)
    # u, s, vt = np.linalg.svd(a=w_)
    # w_star = svd_reconstruct(w_, 90)
    # print(np.linalg.norm(w_ - w_star))

    # print('svd rank vs norm loss for last block in the model')
    # for f_i, f in enumerate(model.module.blocks[3].flows):
    #     print(f'flow {f_i}')
    #     w = f.invconv.weight[:, :, 0, 0].detach().cpu().numpy()
    #     for r in np.arange(2, w.shape[0] + 1):
    #         w_star = svd_reconstruct(w, r)
    #         print(f'r={r},norm = {np.linalg.norm(w - w_star)}')
    # TODO
    # replace w in each flow by w_ reconstruct
    # measure difference visually and then numerically (numerical measures for nf quality , bits of info i think ??)

    # experiment 1 :
    # rank_norm_experiment1(model)
    # full_rank_encode_decode(model, None)
