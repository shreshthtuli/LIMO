from utils import *
from models import *
import torch
import pytorch_lightning as pl
import numpy as np
import argparse
from scipy.stats import linregress


device = "cpu" #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(1)

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['vae', 'tran', 'vae-baseline', 'vae-smi'], default='vae')
parser.add_argument('--prop', choices=['logp', 'penalized_logp', 'qed', 'sa', 'binding_affinity', 'toxicity'], default='binding_affinity')
parser.add_argument('--num_mols', type=int, default=10000)
parser.add_argument('--autodock_executable', type=str, default='./autodock_gpu_64wi')
parser.add_argument('--protein_file', type=str, default='1err/1err.maps.fld')
args = parser.parse_args()

try:
    if args.model == 'vae':
        vae = VAE(max_len=dm.dataset.max_len, vocab_len=len(dm.dataset.symbol_to_idx), latent_dim=1024, embedding_dim=64).to(device)
    elif args.model == 'vae-baseline':
        vae = VAE(max_len=dm.dataset.max_len, vocab_len=len(dm.dataset.symbol_to_idx), latent_dim=1024, embedding_dim=64).to(device)
    elif args.model == 'vae-smi':
        vae = VAE(max_len=dm.dataset.max_len + 100, vocab_len=len(dm.dataset.symbol_to_idx), latent_dim=1024, embedding_dim=64).to(device)
    else:
        vae = Transformer(max_len=dm.dataset.max_len, vocab_len=len(dm.dataset.symbol_to_idx), 
          latent_dim=1024, embedding_dim=64).to(device)
except NameError:
    raise Exception('No dm.pkl found, please run preprocess_data.py first')
vae.load_state_dict(torch.load(f'{args.model}.pt'))
vae.eval()

def generate_training_mols(num_mols, prop_func):
    with torch.no_grad():
        z = torch.randn((num_mols, 1024), device=device) if 'vae' in args.model else torch.randn((num_mols, vae.max_len, vae.embedding_dim), device=device, requires_grad=True)
        x = torch.exp(vae.decode(z))
        y = torch.tensor(prop_func(x), device=device).unsqueeze(1).float()
    return x[1000:], y[1000:], x[:1000], y[:1000]

def get_data(file, pos, neg, dm):
    dataset = ToxDataset(file, dm)
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
    x = torch.cat(x, 0)
    test_x = torch.cat(test_x, 0)
    test_y = torch.tensor(test_y).to(device).unsqueeze(1)
    # oversample = SMOTE(sampling_strategy=0.5)
    # # under = RandomUnderSampler(sampling_strategy=1.0)
    # x,y = oversample.fit_resample(x.cpu(), y)
    # # x,y = under.fit_resample(x,y)
    y = torch.tensor(y).to(device).unsqueeze(1).float()
    return x, y, test_x, test_y

def get_vae_out(mols):
    with torch.no_grad():
        x = vae.forward(mols.to(device))[0]
    return torch.exp(x)

props = {'logp': one_hots_to_logp, 
         'penalized_logp': one_hots_to_penalized_logp, 
         'qed': one_hots_to_qed, 
         'sa': one_hots_to_sa,
         'binding_affinity': lambda x: one_hots_to_affinity(x, args.autodock_executable, args.protein_file)}

x, y, test_x, test_y = generate_training_mols(args.num_mols, props[args.prop]) if args.prop != 'toxicity' else get_data('clintox.csv', 50, 200, dm)
x, test_x = x if args.prop != 'toxicity' else get_vae_out(x), test_x if args.prop != 'toxicity' else get_vae_out(test_x)

model = PropertyPredictor(x.shape[1]) if args.prop != 'toxicity' else ToxPredictor(x.shape[1])
dm = PropDataModule(x, y, 100)
trainer = pl.Trainer(accelerator='mps', max_epochs=30, logger=pl.loggers.CSVLogger('logs'), enable_checkpointing=False, callbacks=[pl.callbacks.LearningRateFinder()])
trainer.fit(model, dm)
model.eval()
model = model.to(device)

print(f'property predictor trained, correlation of r = {linregress(model(test_x).detach().cpu().numpy().flatten(), test_y.detach().cpu().numpy().flatten()).rvalue}')
if not os.path.exists('property_models'):
    os.mkdir('property_models')
torch.save(model.state_dict(), f'property_models/{args.model}/{args.prop}.pt')