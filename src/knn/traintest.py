"""
Train and test KNN classifier. Uses KNN from sklearn library.
"""
import numpy as np
import pandas as pd

import os

import sklearn.metrics as mt
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import argparse

import sys
sys.path.append("src")

from utils import read_data

import warnings
warnings.filterwarnings("ignore")

###############################################################################
#                            Arguments definition                             #
###############################################################################
desc="Train and test KNN."
parser = argparse.ArgumentParser(description=desc)
parser.add_argument("--data", dest="dat", default="data/clustered/",
                    help="pfam data directory (data/clustered/ by default)")
parser.add_argument("--embeddings", dest="emb", default="data/embeddings/",
                    help="embeddings directory (data/embeddings/ by default)")
parser.add_argument("--train", dest="train", default="train",
                    help="partition used for training (train by default)")
parser.add_argument("--dev", dest="dev", default="dev",
                    help="partition used for dev (dev by default)")
parser.add_argument("--test", dest="test", default="test",
                    help="partition used for testing (test by default)")
parser.add_argument("--results", dest="res", default="results/",
                    help="summaries directory (results/ created by default)")

args = parser.parse_args()

if not os.path.isdir(args.res):
    os.mkdir(args.res)

###############################################################################
#                                  Load data                                  #
###############################################################################
# Load dev data
print("Loading dev data...")
dev = read_data(f"{args.dat}{args.dev}/")
dev.reset_index(inplace=True, drop=True)
dev.set_index("sequence_name", inplace=True)
dev_data=np.load(f"{args.emb}{args.dev}.npy",allow_pickle=True).item()

# Load train data
print("Loading train data...")
train = read_data(f"{args.dat}{args.train}/")
train.reset_index(inplace=True, drop=True)
train.set_index("sequence_name", inplace=True)
train_data=np.load(f"{args.emb}{args.train}.npy",allow_pickle=True).item()

print("Merging dev and train...")
for prname in list(dev_data.keys()):
    train_data[prname]=dev_data[prname]

train=pd.concat([train,dev])

del dev, dev_data

train_val=list(map(train_data.get,train.index.values))
train_labels=list(map(train.family_accession.get,train.index.values))

del train, train_data

###############################################################################
#                                  Training                                   #
###############################################################################
print("KNN fit with train+dev embbedding")
knn_model = KNeighborsClassifier(n_neighbors=1, n_jobs=-1, algorithm='brute')
knn_model.fit(train_val, train_labels)

del train_val, train_labels

###############################################################################
#                                   Testing                                   #
###############################################################################
# Load test data
print("Loading test data...")
test = read_data(f"{args.dat}{args.test}/")
test.reset_index(inplace=True, drop=True)
test.set_index("sequence_name", inplace=True)
test_data=np.load(f"{args.emb}{args.test}.npy",allow_pickle=True).item()

test_ind=list(test.index.values)
test_val=list(map(test_data.get,test_ind))
test_labels=list(map(test.family_accession.get,test_ind))

del test, test_data

print("Testing...")
test_pred=knn_model.predict(test_val)

a = mt.accuracy_score(test_labels, test_pred)
print("Error rate:   {0:6.2f}".format((1-a)*100) + "%")
print("Total errors: {0:6d}".format(sum(1-(np.array(test_pred)==
                                           np.array(test_labels)))))
with open(f"{args.res}predicts_knn.csv", 'w') as f:
    f.write("sequence_name,predict,ground_truth\n")
    for i in tqdm(range(len(test_ind))):
        f.write(f"{test_ind[i]},{test_pred[i]},{test_labels[i]}\n")
