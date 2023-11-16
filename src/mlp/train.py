"""
Train MLP classifier. Uses MLP from pytorch library.
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch as tr
from torchmlp import MLP
from torch.utils.data import DataLoader
import sklearn.metrics as mt
import argparse

import sys
sys.path.append("src")

from utils import read_data

###############################################################################
#                            Arguments definition                             #
###############################################################################
desc="Train MLP."
parser = argparse.ArgumentParser(description=desc)
parser.add_argument("--data", dest="dat", default="data/clustered/",
                    help="pfam data directory (data/clustered/ by default)")
parser.add_argument("--embeddings", dest="emb", default="data/embeddings/",
                    help="embeddings directory (data/embeddings/ by default)")
parser.add_argument("--train", dest="train", default="train",
                    help="partition used for training (train by default)")
parser.add_argument("--dev", dest="dev", default="dev",
                    help="partition used for dev (dev by default)")
parser.add_argument("--categories", dest="cat", default="data/categories.txt",
                    help="categories file (data/categories.txt by default)")
parser.add_argument("--models", dest="mod", default="models/",
                    help="resulting models directory (models/ created by "
                    "default)")
parser.add_argument("-n", type=int, default=1, 
                    help="number of models to train (1 by default)")
parser.add_argument("--init", type=int, default=0,
                    help="Init model to train (used to resume training, 0 by "
                    "default)")

args = parser.parse_args()

LAYERS=(500, 100, 100, 1000)
LR = 1e-3
DEVICE = "cuda"
EPOCHS = 1000
PATIENCE = 5
BATCH_SIZE = 32

categories = [item.strip() for item in open(args.cat)]

if not os.path.isdir(args.mod):
    os.mkdir(args.mod)

###############################################################################
#                                  Load data                                  #
###############################################################################
# Load train data
print("Loading train data...")
train = read_data(f"{args.dat}{args.train}/")
train.reset_index(inplace=True, drop=True)
train.set_index("sequence_name", inplace=True)
train_vals=np.load(f"{args.emb}{args.train}.npy", allow_pickle=True).item()
train_data=[[tr.tensor(train_vals[i]),
             tr.tensor(categories.index(train.loc[i].family_id)),
             i] for i in tqdm(train.index, desc="Creating train DataLoader",
                              leave=False)
           ]
del train_vals

emb_size=train_data[0][0].shape[0]

train_dl=DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
del train, train_data

# Load dev data
print("Loading dev data...")
dev = read_data(f"{args.dat}{args.dev}/")
dev.reset_index(inplace=True, drop=True)
dev.set_index("sequence_name", inplace=True)
dev_vals=np.load(f"{args.emb}{args.dev}.npy", allow_pickle=True).item()
dev_data=[[tr.tensor(dev_vals[i]),
           tr.tensor(categories.index(dev.loc[i].family_id)),
           i] for i in tqdm(dev.index, desc="Creating dev DataLoader",
                            leave=False)
         ]
del dev_vals
dev_dl=DataLoader(dev_data, batch_size=BATCH_SIZE, shuffle=True)
del dev, dev_data

print("\n                           MLP fit                           ")
print("-------------------------------------------------------------")

###############################################################################
#                                  Training                                   #
###############################################################################
for model in range(args.init, args.n):
    filename = f"{args.mod}MLP_{model}.pk"
    summary  = f"{args.mod}MLP_{model}_summary.csv"
    if os.path.exists(filename):
        net = tr.load(filename)
        with open(summary, 'r') as s:
            last_sum=s.readlines()[-1].split(',')
            INIT_EP  = int(last_sum[0])+1
            best_err = float(last_sum[4])
            counter  = int(last_sum[5])
    else:
        net = MLP(LAYERS, emb_size, len(categories), device=DEVICE, lr=LR)
        with open(summary, 'w') as s:
            s.write("Ep,Train loss,Dev Loss,Dev error,Best error,Counter\n")
            INIT_EP, counter, best_err = 0, 0, 999.0

    if counter >= PATIENCE:
        continue

    for j in range(INIT_EP, EPOCHS):
        print(f"Epoch: {j}, Model: {model}, Counter: {counter}")
        tr_ls = net.fit(train_dl)
        dv_ls, dev_err, _, _, _ = net.pred(dev_dl)
        # early stop
        if dev_err < best_err:
            best_err = dev_err
            tr.save(net, filename)
            counter = 0
            msg=" - Saved model."
        else:
            counter += 1
            msg=""

        with open(summary, 'a') as s:
            s.write(f"{j},{tr_ls},{dv_ls},{dev_err},{best_err},{counter}\n")
        print(f"{j}: dev error = {dev_err:.3f}{msg}")
        if counter >= PATIENCE:
            break

print("\n\nFinished succesfully")
