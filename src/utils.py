"""
Utility functions for protein models.
"""

import os
import numpy as np
import pandas as pd

def read_data(path):
    """
    Adapted from https://www.kaggle.com/code/petersarvari/protcnn-fast
    """
    shards = []
    for fn in os.listdir(path):
        with open(os.path.join(path, fn)) as f:
            shards.append(pd.read_csv(f, index_col=None))
    return pd.concat(shards)

