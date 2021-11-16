import numpy as np
import torch


def svd_decomp_trunc(w, r):
    u, s, vt = np.linalg.svd(w)
    return u[:, :r], s[:r], vt[:r, :]
    # if r <= u.shape[0]:
    #     m1 = np.matmul(np.diag(s[:r]), vt[:r, :])
    #     w_star = np.matmul(u[:, :r], m1)
    #     return w_star, u, s, vt
    # else:
    #     raise ValueError('r > u num-rows')


def pinv(w, r):
    u, s, vt = np.linalg.svd(w)
    m1 = np.matmul(np.transpose(vt[:r, :]), np.diag(1.0 / s[:r]))
    w_inv = np.matmul(m1, np.transpose(u[:, :r]))
    # assert
    # I_ = np.matmul(w,w_inv)
    # norm_ = np.linalg.norm(np.eye(w.shape[0])-I_)
    # print(f'norm_ for inv assert = {norm_}')
    return w_inv


def tensor_stat(input_tensor):
    return {'mean': torch.mean(input_tensor).item(), 'max': torch.max(input_tensor).item(),
            'min': torch.min(input_tensor).item()}
