"""
Test MLP classifier.
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
desc="Test MLP."
parser = argparse.ArgumentParser(description=desc)
parser.add_argument("--data", dest="dat", default="data/clustered/",
                    help="pfam data directory (data/clustered/ by default)")
parser.add_argument("--embeddings", dest="emb", default="data/embeddings/",
                    help="embeddings directory (data/embeddings/ by default)")
parser.add_argument("--test", dest="test", default="test",
                    help="partition used for testing (test by default)")
parser.add_argument("--categories", dest="cat", default="data/categories.txt",
                    help="categories file (data/categories.txt by default)")
parser.add_argument("--models", dest="mod", default="models/",
                    help="trained models directory (models/ by default)")
parser.add_argument("--results", dest="res", default="results/",
                    help="summaries directory (results/ created by default)")

args = parser.parse_args()

LAYERS=(500, 100, 100, 1000)
BATCH_SIZE = 32

categories = [item.strip() for item in open(args.cat)]

if not os.path.isdir(args.res):
    os.mkdir(args.res)

###############################################################################
#                                  Load data                                  #
###############################################################################
# Load test data
print("Loading test data...")
test = read_data(f"{args.dat}{args.test}/")
test.reset_index(inplace=True, drop=True)
test.set_index("sequence_name", inplace=True)
test_vals=np.load(f"{args.emb}{args.test}.npy", allow_pickle=True).item()
test_data=[[tr.tensor(test_vals[i]),
            tr.tensor(categories.index(test.loc[i].family_id)),
            i] for i in tqdm(test.index, desc="Creating DataLoader",
                             leave=False)
          ]
del test_vals
test_dl=DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Run predictions for each model on folder and get ensembled prediction
pred_avg = tr.zeros((len(test_data), len(categories)))
del test, test_data

# load models
models = [md for md in os.listdir(args.mod) if ".pk" in md]

ref, names = [], []
print(f"Testing MLP")
for model, filename in enumerate(models):
    print(f"Loading model {model}...")
    net=tr.load(f"{args.mod}{filename}")
    _, test_errate, pred, ref, names = net.pred(test_dl)
    pred_bin = tr.argmax(pred, dim=1)
    a = mt.accuracy_score(ref, pred_bin)
    print("Error rate:   {0:6.2f}".format((1-a)*100) + "%")
    print("Total errors: {0:6d}".format(sum(1-(np.array(pred_bin)==
                                               np.array(ref)))))

    with open(f"{args.res}errors_{model}.csv",'w') as f:
        f.write("sequence_name,predict,ground_truth\n")
        for i in tqdm(range(len(ref)),desc="Saving predictions",leave=False):
            f.write(f"{names[i]},{ref[i]},{pred_bin[i]}\n")
    # k-ensemble score
    pred_avg += pred
print("\nEnsemble score:")
pred_avg_bin = tr.argmax(pred_avg, dim=1)
a = mt.accuracy_score(ref, pred_avg_bin)
print("Error rate:   {0:6.2f}".format((1-a)*100) + "%")
print("Total errors: {0:6d}".format(sum(1-(np.array(pred_avg_bin)==
                                    np.array(ref)))))
with open(f"{args.res}errors_ensamble.csv",'w') as f:
    f.write("sequence_name,predict,ground_truth\n")
    for i in tqdm(range(len(ref)),desc="Saving predictions",leave=False):
        f.write(f"{names[i]},{ref[i]},{pred_avg_bin[i]}\n")
print("\n\nFinished succesfully")
