from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import torch
from torchvision import datasets, transforms
import argparse
import os


def calculate_coeff(data, log=False, plot=False, show=False):
    data = np.array(data)
    data_clean = data[np.all(~np.isnan(data), axis=1)]
    ind_sort = np.argsort(data_clean[:,0])
    data_clean = data_clean[ind_sort]

    d, ll = tuple(zip(*data_clean))

    d = np.array(d).reshape(-1,1)
    ll = np.array(ll)
    
    if log:
        d = np.log(d)
    regr = linear_model.LinearRegression()
    regr.fit(d, ll)
    ll_pred = regr.predict(d)

    if plot:
        plt.plot(d, regr.predict(d), label="prediction", c='r', alpha=0.5)
        plt.plot(d, ll, 'o-', alpha=0.5)
    if show:
        plt.show()
        
    return regr.coef_[0]


def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]


def summary(ll_path, epochs):
    results = list()
    print((('dim', ('inds',) + (('epoch', ),))))
    for epoch in list(range(10, epochs)):
        fnames = sorted(glob(os.path.join(ll_path, f"ll_*_{epoch}.txt")))
        points = []
        for f in fnames:
            txt = pd.read_csv(f, sep=" ", header=None)
            noise, pz, logdet, _, _ = txt.mean()
            points.append([noise, -(pz + logdet)])
        points = np.array(points)

        all_deltas_inds = [x for x in powerset(list(range(8))) if len(x) >= 2]
        for inds in all_deltas_inds:
            dim = calculate_coeff(points[np.array(inds), :], log=True, plot=False)
            results.append((dim, tuple(inds) + ((epoch, ),)))

    results.sort()
    for r in results:
        print(r)

    print('----------')
    print('deltas:')
    for i in range(points.shape[0]):
        print(f"{i}: {points[i][0]}")

#summary(f"/home/rm360179/glow-pytorch/results_ffhq_50/ll", 75)
#summary(f"/home/rm360179/glow-pytorch/ll", 40)

parser = argparse.ArgumentParser()
parser.add_argument("-n", help="number of epochs", required=True)
parser.add_argument("-ll", help="path to ll", required=True)
args = parser.parse_args()

ll_path = args.ll
max_epochs = int(args.n)
summary(ll_path, max_epochs)
