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

args = parser.parse_args()

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
test = read_data(f"{args.dat}{args.test}/")
test.reset_index(inplace=True, drop=True)
test.set_index("sequence_name", inplace=True)
test_data=np.load(f"{args.emb}{args.test}.npy",allow_pickle=True).item()

test_val=list(map(test_data.get,list(test.index.values)))
test_labels=list(map(test.family_accession.get,list(test.index.values)))

del test, test_data

test_pred=knn_model.predict(test_val)

a = mt.accuracy_score(test_labels, test_pred)
print("Error rate:   {0:6.2f}".format((1-a)*100) + "%")
print("Total errors: {0:6d}".format(sum(1-(np.array(test_pred)==
                                           np.array(test_labels)))))
