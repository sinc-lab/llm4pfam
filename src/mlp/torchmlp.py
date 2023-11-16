"""
Pytorch MLP class definition.
"""
import torch as tr
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class MLP(nn.Module):
    def __init__(self, layers, emb_size, nclasses, device="cuda", lr=1e-3):

        nn.Module.__init__(self)

        self.layers = layers
        self.emb_size = emb_size
        self.nclasses = nclasses
        self.device = device
        self.lr = lr

        self.net = nn.Sequential()
        self.net.add_module("input", nn.Linear(emb_size, layers[0]))
        self.net.add_module(f"act0", nn.Identity())
        for i in range(len(layers)-1):
            self.net.add_module(f"hidden{i}",nn.Linear(layers[i],layers[i+1]))
            self.net.add_module(f"act{i+1}",nn.Identity())
        self.net.add_module("output",nn.Linear(layers[-1],nclasses))
        self.net.to(self.device)

        self.loss = nn.CrossEntropyLoss()
        self.optim = tr.optim.Adam(self.net.parameters(), lr=self.lr)

    def forward(self, emb):
        y = self.net(emb.type(tr.FloatTensor).to(self.device))
        return y

    def fit(self, dataloader):
        avg_loss = 0
        self.train()
        self.optim.zero_grad()

        for emb, yref, _ in tqdm(dataloader, leave=False):
            ypred = self(emb.to(self.device))
            yref = yref.to(self.device)

            loss = self.loss(ypred, yref)
            loss.backward()
            avg_loss += loss.item()
            self.optim.step()
            self.optim.zero_grad()

        avg_loss /= len(dataloader)
        return avg_loss

    def pred(self, dataloader):
        test_loss = 0
        pred, refs, names = [], [], []
        self.eval()

        for emb, yref, name in tqdm(dataloader, leave=False):
            with tr.no_grad():
                ypred = self(emb.to(self.device))
                yref = yref.to(self.device)
                test_loss += self.loss(ypred, yref).item()

            names += name
            pred.append(ypred.detach().cpu())
            refs.append(yref.cpu())

        pred = tr.cat(pred)
        pred_bin = tr.argmax(pred, dim=1)
        refs = tr.cat(refs)

        test_loss /= len(dataloader)
        acc = accuracy_score(refs, pred_bin)
        return test_loss, 1-acc, pred, refs, names
