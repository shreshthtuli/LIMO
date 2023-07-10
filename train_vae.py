from utils import *
from models import *
import torch
import pytorch_lightning as pl
torch.cuda.empty_cache()


if __name__ == '__main__':
    vae = VAE(max_len=dm.dataset.max_len, vocab_len=len(dm.dataset.symbol_to_idx), 
            latent_dim=1024, embedding_dim=64)
    trainer = pl.Trainer(accelerator='mps', max_epochs=18, logger=pl.loggers.CSVLogger('logs'), enable_checkpointing=False)
    print('Training..')
    trainer.fit(vae, dm)
    print('Saving..')
    torch.save(vae.state_dict(), 'vae.pt')