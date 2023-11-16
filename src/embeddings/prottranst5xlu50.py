from tqdm import tqdm
import os
import pickle
import numpy as np
import torch as tr
import argparse

from transformers import T5Tokenizer, T5EncoderModel
import re

import sys
sys.path.append("src")

from utils import read_data

###############################################################################
#                            Arguments definition                             #
###############################################################################
desc="Precompute protTrans T5-XL-U50 to speed up training."
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

tr.hub.set_dir(args.cache)

###############################################################################
#                                Load embedder                                #
###############################################################################
print("Loading tokenizer...")
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc",
                                        cache_dir=args.cache,
                                        do_lower_case=False)
print("Loading model...")
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc",
                                       cache_dir=args.cache)
model.full() if args.device=='cpu' else model.half()
model = model.to(args.device)
model = model.eval()

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

    seq = data.iloc[item].sequence
    seq_len=len(seq)
    seq=" ".join(list(re.sub(r"[UZOB]", "X", seq)))
    label = data.iloc[item].family_id

    ids = tokenizer.batch_encode_plus([seq], add_special_tokens=True,
                                      padding="longest")
    input_ids = tr.tensor(ids['input_ids']).to(args.device)
    attention_mask = tr.tensor(ids['attention_mask']).to(args.device)

    try:
        with tr.no_grad():
            emb = model(input_ids=input_ids,
                        attention_mask=attention_mask
                        ).last_hidden_state[:,:seq_len,:].detach().cpu()
    except:
        print(seq_name, seq_len)
        raise

    if args.premb:
        emb = emb.permute(0,2,1)
        pickle.dump([emb.half(), label], open(out_file, "wb"))

    if args.ppemb:
        results[seq_name]=np.array(emb[0].mean(dim=0))

if args.ppemb:
    np.save(f"{args.emb}{args.part}.npy",results)
