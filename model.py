import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

embedding_dim = 128
hidden_dim = 1024
vocab_size = len(word2id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Encoderの定義
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size, z_dim=128):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        # 埋め込み層(ベクトル表現に変換)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=word2id["<pad>"])
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        
        # VAEを追加
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.log_var = nn.Linear(hidden_dim, z_dim)
        self.out = nn.Linear(z_dim, hidden_dim)

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(log_var/2)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            std = torch.exp(log_var/2)
            eps = torch.randn_like(std)
            return mu + eps * std

    def encode(self, index):
        embedding = self.word_embeddings(index)

        if embedding.dim() == 2:
            embedding = torch.unsqueeze(embedding, 1)
        
        _, state = self.gru(embedding, torch.zeros(1, self.batch_size, self.hidden_dim, device=device))

        return self.mu(state), self.log_var(state)

    def forward(self, index):
        mu, log_var = self.encode(index)
        z = self.reparameterize(mu, log_var)
        state = self.out(z)
        
        return mu, log_var, z, state

# Decoderの定義
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=word2id["<pad>"])
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, index, state):
        embedding = self.word_embeddings(index)

        if embedding.dim() == 2:
            embedding = torch.unsqueeze(embedding, 1)

        gruout, state = self.gru(embedding, state)
        output = self.output(gruout)

        return output, state