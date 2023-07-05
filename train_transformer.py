from utils import *
from models import *
import torch
import pytorch_lightning as pl

tran = Transformer(max_len=dm.dataset.max_len, vocab_len=len(dm.dataset.symbol_to_idx), 
          latent_dim=1024, embedding_dim=64)
trainer = pl.Trainer(gpus=0, max_epochs=18, logger=pl.loggers.CSVLogger('logs'), enable_checkpointing=False)
print('Training..')
trainer.fit(tran, dm)
print('Saving..')
torch.save(tran.state_dict(), 'tran.pt')