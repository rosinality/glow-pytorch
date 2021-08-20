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


def summary(ll_path, epochs, deltas_count, deltas_total):
    assert 2 <= deltas_count <= deltas_total
    #results = list()
    #results = [list() for _ in range(9)]
    #print((('dim', ('inds',) + (('epoch', ),))))
    #results = [pd.DataFrame(columns=['epochs', 'avg_delta', 'dim']) for _ in range(deltas+1)]
    dims = list()
    #results = [[list() for _ in range(epochs)] for _ in range(deltas_count+1)]


    for epoch in list(range(1, epochs+1)):
        dims.append(list())
        fnames = sorted(glob(os.path.join(ll_path, f"ll_*_{epoch}.txt")))
        points = []
        for f in fnames:
            txt = pd.read_csv(f, sep=" ", header=None)
            noise, pz, logdet, _, _ = txt.mean()
            points.append([noise, -(pz + logdet)])
        assert deltas_total == len(points)
        points = np.array(points)
        deltas = list()
        for first_delta in range(0, deltas_total-deltas_count+1):
            inds = np.arange(first_delta, first_delta+deltas_count)
            #print(points)
            #print(inds)
            avg_delta = sum(points[ind][0] for ind in inds)/len(inds)
            dim = calculate_coeff(points[inds, :], log=True, plot=False)
            dims[epoch-1].append(dim)
            deltas.append(avg_delta)

    return np.array(dims), np.array(deltas), np.arange(1, epochs+1)



#summary(f"/home/rm360179/glow-pytorch/results_ffhq_50/ll", 75)
#summary(f"/home/rm360179/glow-pytorch/ll", 40)

parser = argparse.ArgumentParser()
parser.add_argument("-n", help="number of epochs", required=True)
parser.add_argument("-ll", help="path to ll", required=True)
parser.add_argument("-d", help="number of deltas", required=True)
parser.add_argument("-ds", help="size of deltas subsets", required=True)
args = parser.parse_args()

deltas_subset_size = 2
ll_path = args.ll
max_epochs = int(args.n)
deltas_total = int(args.d)
deltas_subset_size=int(args.ds)
dims, deltas, epochs = summary(ll_path, max_epochs, deltas_subset_size, deltas_total)

print(deltas)
print(dims)
#print(np.vstack((deltas, dims)))
#plt.yticks(np.array(deltas))
#plt.xticks(np.arange(deltas_total-deltas_subset_size))
plt.imshow(dims)
plt.savefig(f'plots/plot.png')

#plt.imshow(results[i]['dim'].to_numpy().reshape((max_epochs, deltas-i)))
    #deltas = results[i]['avg_delta'].to_numpy()
    #plt.xticks(np.arange(deltas-i))
    #plt.yticks(np.arange(1, max_epochs+1))
    #plt.imshow(results[i]['epochs'].to_numpy().reshape((max_epochs, deltas-i)), interpolation='bilinear')
    #print(results[i]['epochs'].to_numpy())
