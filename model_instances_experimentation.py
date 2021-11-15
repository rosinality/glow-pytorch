import matplotlib.pyplot as plt
import torch
import torchvision.utils
from torch import nn

from model import Glow
from train import calc_z_shapes


def rank_norm(model):
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
    y= []
    for k in rank_norm_dict.keys():
        x.append(k)
        y.append(rank_norm_dict[k][0]/rank_norm_dict[k][1])
    plt.plot(x,y)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    affine = False
    conv_lu = False
    model_single = Glow(3, n_flow, n_block, affine=affine, conv_lu=False)
    model = nn.DataParallel(model_single)
    # checkpoint = torch.load('checkpoint')
    # # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html
    # print(type(checkpoint))
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(torch.load('checkpoint_2021-11-07T14:25:53.737008/model_006000.pt'))
    # print(model.module.blocks[3].flows[0].invconv.weight.size())
    # print(type(model.module.blocks[3].flows[0].invconv.weight[:, :, 0, 0].detach().cpu().numpy().shape))
    # w_ = model.module.blocks[3].flows[0].invconv.weight[:, :, 0, 0].detach().cpu().numpy()
    z_sample = []
    z_shapes = calc_z_shapes(3, img_size, n_flow, n_block)
    print('generating sample image')
    gen_sample_img_path = './gen_sample9.png'
    for z in z_shapes:
        z_new = torch.randn(n_sample, *z) * temp
        z_sample.append(z_new.to(device))
    gen_sample = model.module.reverse(z_sample).cpu().data
    torchvision.utils.save_image(tensor=gen_sample, fp=gen_sample_img_path, normalize=True,
                                 nrow=10,
                                 range=(-0.5, 0.5))

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
    rank_norm(model)
