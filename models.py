import pytorch_lightning as pl
import torch, math
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class VAE(pl.LightningModule):
    def __init__(self, max_len, vocab_len, latent_dim, embedding_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.vocab_len = vocab_len
        self.embedding = nn.Embedding(vocab_len, embedding_dim, padding_idx=0)
        self.encoder = nn.Sequential(nn.Linear(max_len * embedding_dim, 2000),
                                     nn.ReLU(),
                                     nn.Linear(2000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, latent_dim * 2))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 2000),
                                     nn.ReLU(),
                                     nn.Linear(2000, max_len * vocab_len))
        
    def encode(self, x):
        x = self.encoder(self.embedding(x).view((len(x), -1))).view((-1, 2, self.latent_dim))
        mu, log_var = x[:, 0, :], x[:, 1, :]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std, mu, log_var
    
    def decode(self, x):
        return F.log_softmax(self.decoder(x).view((-1, self.max_len, self.vocab_len)), dim=2).view((-1, self.max_len * self.vocab_len))
    
    def forward(self, x):
        z, mu, log_var = self.encode(x)
        return self.decode(z), z, mu, log_var
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        return {'optimizer': optimizer}
    
    def loss_function(self, pred, target, mu, log_var, batch_size, p):
        nll = F.nll_loss(pred, target)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / (batch_size * pred.shape[1])
        return (1 - p) * nll + p * kld, nll, kld
    
    def training_step(self, train_batch, batch_idx):
        out, z, mu, log_var = self(train_batch)
        p = 0.1
        loss, nll, kld = self.loss_function(out.reshape((-1, self.vocab_len)), train_batch.flatten(), mu, log_var, len(train_batch), p)
        self.log('train_loss', loss)
        self.log('train_nll', nll)
        self.log('train_kld', kld)
        return loss
        
    def validation_step(self, val_batch, batch_idx):
        out, z, mu, log_var = self(val_batch)
        loss, nll, kld = self.loss_function(out.reshape((-1, self.vocab_len)), val_batch.flatten(), mu, log_var, len(val_batch), 0.5)
        self.log('val_loss', loss)
        self.log('val_nll', nll)
        self.log('val_kld', kld)
        self.log('val_mu', torch.mean(mu))
        self.log('val_logvar', torch.mean(log_var))
        return loss
    
class VAESMI(pl.LightningModule):
    def __init__(self, max_len, vocab_len, latent_dim, embedding_dim, smi_sim = 100):
        super(VAESMI, self).__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.vocab_len = vocab_len
        self.embedding = nn.Embedding(vocab_len, embedding_dim, padding_idx=0)
        self.encoder = nn.Sequential(nn.Linear(max_len * embedding_dim, 2000),
                                     nn.ReLU(),
                                     nn.Linear(2000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, latent_dim * 2))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 1000),
                                     nn.BatchNorm1d(1000),
                                     nn.ReLU(),
                                     nn.Linear(1000, 2000),
                                     nn.ReLU(),
                                     nn.Linear(2000, max_len * vocab_len))
        
    def encode(self, x):
        x, smi = x
        print(x.shape, smi.shape)
        x = self.encoder(self.embedding(x).view((len(x), -1))).view((-1, 2, self.latent_dim))
        mu, log_var = x[:, 0, :], x[:, 1, :]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std, mu, log_var
    
    def decode(self, x):
        return F.log_softmax(self.decoder(x).view((-1, self.max_len, self.vocab_len)), dim=2).view((-1, self.max_len * self.vocab_len))
    
    def forward(self, x):
        z, mu, log_var = self.encode(x)
        return self.decode(z), z, mu, log_var
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        return {'optimizer': optimizer}
    
    def loss_function(self, pred, target, mu, log_var, batch_size, p):
        nll = F.nll_loss(pred, target)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / (batch_size * pred.shape[1])
        return (1 - p) * nll + p * kld, nll, kld
    
    def training_step(self, train_batch, batch_idx):
        out, z, mu, log_var = self(train_batch)
        p = 0.1
        loss, nll, kld = self.loss_function(out.reshape((-1, self.vocab_len)), train_batch.flatten(), mu, log_var, len(train_batch), p)
        self.log('train_loss', loss)
        self.log('train_nll', nll)
        self.log('train_kld', kld)
        return loss
        
    def validation_step(self, val_batch, batch_idx):
        out, z, mu, log_var = self(val_batch)
        loss, nll, kld = self.loss_function(out.reshape((-1, self.vocab_len)), val_batch.flatten(), mu, log_var, len(val_batch), 0.5)
        self.log('val_loss', loss)
        self.log('val_nll', nll)
        self.log('val_kld', kld)
        self.log('val_mu', torch.mean(mu))
        self.log('val_logvar', torch.mean(log_var))
        return loss

class Transformer(pl.LightningModule):
    def __init__(self, max_len, vocab_len, latent_dim, embedding_dim):
        super(Transformer, self).__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.vocab_len = vocab_len
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_len, embedding_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embedding_dim, 0.1)
        encoder_layers = TransformerEncoderLayer(d_model=embedding_dim, nhead=8, dim_feedforward=1000, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers = TransformerDecoderLayer(d_model=embedding_dim, nhead=8, dim_feedforward=1000, dropout=0.1)
        self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
        self.fcn = nn.Sequential(nn.Linear(max_len * embedding_dim, max_len * vocab_len))
        
    def encode(self, x):
        x = x * math.sqrt(self.embedding_dim)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return x
    
    def decode(self, z):
        out = self.transformer_decoder(z,z)
        out = self.fcn(out.flatten(start_dim=1))
        out = out.view((-1, self.max_len, self.vocab_len))
        return F.log_softmax(out, dim=2).view((-1, self.max_len * self.vocab_len))
    
    def forward(self, x):
        x = self.embedding(x)
        z = self.encode(x)
        return self.decode(z), z
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.0001)
        return {'optimizer': optimizer}
    
    def loss_function(self, pred, target):
        nll = F.nll_loss(pred, target)
        return nll
    
    def training_step(self, train_batch, batch_idx):
        out, _ = self(train_batch)
        loss = self.loss_function(out.reshape((-1, self.vocab_len)), train_batch.flatten())
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, val_batch, batch_idx):
        out, _ = self(val_batch)
        loss = self.loss_function(out.reshape((-1, self.vocab_len)), val_batch.flatten())
        self.log('val_loss', loss)
        return loss
    
class PropertyPredictor(pl.LightningModule):
    def __init__(self, in_dim, learning_rate=0.001):
        super(PropertyPredictor, self).__init__()
        self.learning_rate = learning_rate
        self.fc = nn.Sequential(nn.Linear(in_dim, 1000),
                                nn.ReLU(),
                                nn.Linear(1000, 1000),
                                nn.ReLU(),
                                nn.Linear(1000, 1))
        
    def forward(self, x):
        return self.fc(x)
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def loss_function(self, pred, real):
        return F.mse_loss(pred, real)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_function(out, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_function(out, y)
        self.log('val_loss', loss)
        return loss

class ToxPredictor(PropertyPredictor):
    def __init__(self, in_dim, learning_rate=5.0e-4):
        super(ToxPredictor, self).__init__(in_dim)
        self.learning_rate = learning_rate
    
    def loss_function(self, pred, real):
        return F.binary_cross_entropy(pred, real)

##### DL utils #####

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[:, pos:pos+x.size(1), :]
        return self.dropout(x)
