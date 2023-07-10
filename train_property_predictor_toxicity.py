from utils import *
from models import *
import torch
import pytorch_lightning as pl
import numpy as np
import argparse
from scipy.stats import linregress
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pl.seed_everything(42)

try:
    vae = VAE(max_len=dm.dataset.max_len, vocab_len=len(dm.dataset.symbol_to_idx), latent_dim=1024, embedding_dim=64).to(device)
except NameError:
    raise Exception('No dm.pkl found, please run preprocess_data.py first')
vae.load_state_dict(torch.load('vae.pt'))
vae.eval()

def get_data(file, pos, neg):
    dataset = ToxDataset(file)
    x = []
    y = []
    test_x = []
    test_y = []
    num_pos = 0
    for mol, label in dataset:
        if (label == 1):
            if (num_pos < pos):
                num_pos += 1
                x.append(mol.unsqueeze(0))
                y.append(label)
            else:
                test_x.append(mol.unsqueeze(0))
                test_y.append(label)
        else:
            if (len(y) - num_pos < neg):
                x.append(mol.unsqueeze(0))
                y.append(label)
            else:
                test_x.append(mol.unsqueeze(0))
                test_y.append(label)
        # print(mol.shape)
    x = torch.cat(x, 0)
    test_x = torch.cat(test_x, 0)
    test_y = torch.tensor(test_y).to(device).unsqueeze(1).float()
    # oversample = SMOTE(sampling_strategy=0.5)
    # # under = RandomUnderSampler(sampling_strategy=1.0)
    # x,y = oversample.fit_resample(x.cpu(), y)
    # # x,y = under.fit_resample(x,y)
    # x = torch.tensor(x).to(device)
    y = torch.tensor(y).to(device).unsqueeze(1).float()
    print(x.shape)
    print(test_x.shape)
    return x, y, test_x, test_y

def get_vae_out(mols):
    with torch.no_grad():
        x, _, _, _ = vae.forward(mols.to(device))
    return torch.exp(x)

x, y, test_x, test_y = get_data('clintox.csv', 50, 200)
x = get_vae_out(x)
test_x = get_vae_out(test_x)
print(x.shape)
model = ToxPredictor(x.shape[1])
dm = PropDataModule(x, y, 10)
trainer = pl.Trainer(gpus=1, max_epochs=30, logger=pl.loggers.CSVLogger('logs'), enable_checkpointing=False)
# trainer.tune(model, dm)
trainer.fit(model, dm)
model.eval()
model = model.to(device)

print(f'property predictor trained')
# preds = list(model(x[:200].to(device)).detach().cpu().numpy().flatten())
# gold = list(y[:200].detach().cpu().numpy().flatten())
preds = list(model(test_x.to(device)).detach().cpu().numpy().flatten())
gold = list(test_y.detach().cpu().numpy().flatten())
print(preds)
tp, fp, fn, tn = 0, 0, 0, 0
# print(preds)
for idx, _ in enumerate(preds):
    pred = int(_ >= 0.5)
    if (pred == int(gold[idx])):
        if (int(gold[idx])== 1):
            tp += 1
        else:
            tn += 1
    else:
        if (int(gold[idx]) == 1):
            fn += 1
        else:
            fp += 1
print(f'accuracy = {(tp + tn)/ len(preds)}')
try:
    print(f'precision of positive = {(tp)/ (tp + fp)}')
except:
    print(f'precision of positive = {0}')
try:
    print(f'recall of positive = {(tp)/ (tp + fn)}')
except:
    print(f'recall of positive = {0}')
try:
    print(f'precision of negative = {(tn)/ (tn + fn)}')
except:
    print(f'precision of negative = {0}')
try:
    print(f'recall of negative = {(tn)/ (fp + tn)}')
except:
    print(f'recall of negative = {0}')
print(f'roc-auc = {roc_auc_score(gold, preds)}')
if not os.path.exists('property_models'):
    os.mkdir('property_models')
torch.save(model.state_dict(), f'property_models/toxicity.pt')