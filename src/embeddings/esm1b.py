from tqdm import tqdm
import os
import pickle
import numpy as np
import torch as tr
import argparse

import sys
sys.path.append("src")

from utils import read_data

###############################################################################
#                            Arguments definition                             #
###############################################################################
desc="Precompute protTrans ESM1b to speed up training."
parser = argparse.ArgumentParser(description=desc)
parser.add_argument("--data", dest="dat", default="data/clustered/",
                    help="pfam data directory (data/clustered/ by default)")
parser.add_argument("--embeddings", dest="emb", default="data/embeddings/",
                    help="output path (data/embeddings/ created by default)")
parser.add_argument("--cache", default="cache/",
                    help="cache path to save embedders (cache/ created by "
                    "default)")
parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="device (cuda by default)")
parser.add_argument("-p", "--partition", dest="part", required=True,
                    choices=["test", "train", "dev"],
                    help="data partition (required)")
parser.add_argument("--per-protein", dest="ppemb", action='store_true',
                    help="select per-protein embeddings (a vector for each"
                    "protein) needed for mlp and knn")
parser.add_argument("--per-residue", dest="premb", action='store_true',
                    help="select per-residue embeddings (a matrix for each"
                    "protein) needed for cnn")

args = parser.parse_args()

if not (args.ppemb ^ args.premb):
    print("ERROR: Select one of the following options: --per-protein or "
          "--per-residue")
    quit()

if not os.path.isdir(args.emb):
    os.mkdir(args.emb)

if not os.path.isdir(args.cache):
    os.mkdir(args.cache)

max_len = 1022

tr.hub.set_dir(args.cache)

###############################################################################
#                                Load embedder                                #
###############################################################################
print("Loading model...")
model, alphabet = tr.hub.load("facebookresearch/esm:main",
                              "esm1b_t33_650M_UR50S")
model.eval()
model.to(args.device)
batch_converter = alphabet.get_batch_converter()

###############################################################################
#                                  Load data                                  #
###############################################################################
print("Loading data...")
data = read_data(f"{args.dat}{args.part}/")

###############################################################################
#                              Compute embedding                              #
###############################################################################
if args.ppemb:
    results={}
for item in tqdm(range(len(data))):
    seq_name = data.iloc[item].sequence_name
    out_file = f"{args.emb}{seq_name.replace('/', '-')}.pk"
    if os.path.isfile(out_file):
        continue

    # Crop larger domains to a center window
    seq = data.iloc[item].sequence
    label = data.iloc[item].family_id

    center = len(seq)//2
    start = max(0, center - max_len//2)
    end = min(len(seq), center + max_len//2)
    seq = seq[start:end]

    x = [(0, seq)]

    try:
        with tr.no_grad():
            _, _, tokens = batch_converter(x)
            emb = model(tokens.to(args.device), repr_layers=[33],
                        return_contacts=True
                        )["representations"][33].detach().cpu()
    except:
        print(seq_name, len(seq))
        raise

    if args.premb:
        emb = emb.permute(0,2,1)
        pickle.dump([emb.half(), label], open(out_file, "wb"))

    if args.ppemb:
        results[seq_name]=np.array(emb[0].mean(dim=0))

if args.ppemb:
    np.save(f"{args.emb}{args.part}.npy",results)
