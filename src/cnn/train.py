"""
Train CNN classifier. Adapted from
https://github.com/sinc-lab/transfer-learning-pfam/blob/master/train.py
"""
from datetime import datetime
import os
import torch as tr
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import argparse

from dataset import ProtDataset
from dataset import pad_batch, BatchSampler
from tlprotcnn import TLProtCNN

###############################################################################
#                            Arguments definition                             #
###############################################################################
desc="Train CNN model based on ProtCNN."
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

args = parser.parse_args()

LR = 1e-3
DEVICE = "cuda"
EPOCHS = 1000
PATIENCE = 10
BATCH_SIZE = 32
WORKERS = 4

categories = [item.strip() for item in open(args.cat)]

if not os.path.isdir(args.mod):
    os.mkdir(args.mod)

###############################################################################
#                                  Load data                                  #
###############################################################################
# Load train data
train_data = ProtDataset(f"{args.dat}{args.train}/", categories, 
                         emb_path=args.emb)
train_loader = DataLoader(train_data, 
                          batch_sampler=BatchSampler(train_data.get_lengths(), 
                                                     sorted=False, 
                                                     batch_size=BATCH_SIZE), 
                          collate_fn=pad_batch, num_workers=WORKERS)

emb_size=train_data[0][0].shape[1]

# Load dev data
dev_data = ProtDataset(f"{args.dat}{args.dev}/", categories, emb_path=args.emb)
dev_loader = DataLoader(dev_data, batch_size=BATCH_SIZE, collate_fn=pad_batch, 
                        num_workers=WORKERS)

###############################################################################
#                                  Training                                   #
###############################################################################
for nrepeat in range(args.n):
    OUT_DIR = f"{args.mod}results_{str(datetime.now())}/"
    if not os.path.isdir(OUT_DIR):
        os.mkdir(OUT_DIR)
    logger = SummaryWriter(OUT_DIR)

    counter, best_err = 0, 999
    net = TLProtCNN(len(categories), lr=LR, device=DEVICE, logger=logger,
                    emb_size=emb_size)

    for epoch in range(EPOCHS):

        train_loss = net.fit(train_loader)
        dev_loss, dev_err, _, _, _ = net.pred(dev_loader)

        # early stop
        nm_msg=''
        if dev_err < best_err:
            best_err = dev_err
            tr.save(net.state_dict(), f"{OUT_DIR}weights.pk")
            counter = 0
            nm_msg=" - Saved model."
        else:
            counter += 1
            if counter > PATIENCE:
                break
        print(f"{epoch}: train loss {train_loss:.3f}, dev loss {dev_loss:.3f},"
              "dev err {dev_err:.3f}"+nm_msg)
