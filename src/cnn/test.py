"""
Test CNN classifier.
"""
import os
import torch as tr
import sklearn.metrics as mt
import numpy as np
from tqdm import tqdm

from dataset import ProtDataset, pad_batch
from torch.utils.data import DataLoader
from tlprotcnn import TLProtCNN
import argparse

###############################################################################
#                            Arguments definition                             #
###############################################################################
desc="Test CNN model based on ProtCNN."
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

BATCH_SIZE = 32
DEVICE = "cuda"

categories = [item.strip() for item in open(args.cat)]

# trained model weights
models = [f"{d}/weights.pk" for d in os.listdir(args.mod) if "result" in d]

test_data = ProtDataset(f"{args.dat}{args.test}/", categories, args.emb)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=pad_batch)

emb_size=test_data[0][0].shape[1]

if not os.path.isdir(args.res):
    os.mkdir(args.res)

# run predictions for each model on the folder, and get the ensembled prediction
pred_avg = tr.zeros((len(test_data), len(categories)))
ref = tr.zeros((len(test_data), 1))
for k, model in enumerate(models):
    print("load weights from", model)
    net = TLProtCNN(len(categories), device=DEVICE, emb_size=emb_size)
    net.load_state_dict(tr.load(f"{args.mod}{model}"))
    net.eval()
    _, test_errate, pred, ref, names = net.pred(test_loader)

    pred_bin = tr.argmax(pred, dim=1)
    a = mt.accuracy_score(ref, pred_bin)
    print("Error rate:   {0:6.2f}".format((1-a)*100) + "%")
    print("Total errors: {0:6d}".format(sum(1-(np.array(pred_bin)==
                                               np.array(ref)))))

    with open("{}errors_model_{}.csv".format(args.res,k),'w') as f:
        for i in tqdm(range(len(ref))):
            if np.array(pred_bin[i])!=np.array(ref[i]):
                f.write("{},{},{}\n".format(names[i],pred_bin[i],ref[i]))

    # k-ensemble score
    pred_avg += pred

print("\nEnsemble score:")
pred_avg_bin = tr.argmax(pred_avg, dim=1)
a = mt.accuracy_score(ref, pred_avg_bin)
print("Error rate:   {0:6.2f}".format((1-a)*100) + "%")
print("Total errors: {0:6d}".format(sum(1-(np.array(pred_avg_bin)==
                                    np.array(ref)))))

with open("{}errors_ensamble.csv".format(args.res),'w') as f:
    for i in tqdm(range(len(ref))):
        if np.array(pred_avg_bin[i])!=np.array(ref[i]):
            f.write("{},{},{}\n".format(names[i],pred_avg_bin[i],ref[i]))
