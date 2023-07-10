from utils import *
from models import *
import torch
import pytorch_lightning as pl

if __name__ == '__main__':
    tran = Transformer(max_len=dm.dataset.max_len, vocab_len=len(dm.dataset.symbol_to_idx), 
            latent_dim=1024, embedding_dim=64)
    trainer = pl.Trainer(accelerator='mps', max_epochs=18, logger=pl.loggers.CSVLogger('logs'), precision=16, enable_checkpointing=False)
    print('Training..')
    trainer.fit(tran, dm)
    print('Saving..')
    torch.save(tran.state_dict(), 'tran.pt')