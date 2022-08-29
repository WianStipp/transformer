import math
import torch as T
from torch import nn
import torch.nn.functional as F

from transformer.classic import attention
from transformer.classic import feedforward

class Encoder(nn.Module):
  def __init__(self, n_encoder_blocks: int, d_model: int, vocab) -> None:
    super().__init__()
    self.encoder_blocks = [EncoderBlock() for _ in range(n_encoder_blocks)]
    self.embed = Embedding(d_model, vocab)
    self.positional_encoding = lambda: None
  
  def forward(self, X: T.Tensor, mask: T.Tensor):
    E = self.positional_encoding(self.embed(X))
    for encode_block  in self.encoder_blocks:
      E = encode_block(E, mask)
    return E

class EncoderBlock(nn.Module):
  def __init__(self, n_heads: int, d_model: int, d_k: int, d_v: int) -> None:
    self.multihead_attn = attention.MultiHeadAttention(n_heads, d_model, d_k, d_v)
    self.ff = feedforward.Feedforward(d_model)
    
  def forward(self, X: T.Tensor, mask: T.Tensor):
    Z = F.layer_norm(self.multihead_attn(X, X, X) + X)
    return F.layer_norm(self.ff(Z) + Z)

class Embedding(nn.Module):
  def __init__(self, d_model: int, vocab) -> None:
    super().__init__()
    self.wte = nn.Embedding(vocab, d_model)
    self.d_model = d_model

  def forward(self, X):
    return self.wte(X) * (self.d_model ** 0.5)

class PositionalEncoder(nn.Module):
  def __init__(self, d_model: int, dropout: nn.Module, max_length: int = 5000) -> None:
    super().__init__()
    self.d_model = d_model
    self.dropout = dropout
    self.max_length = max_length

    pe = T.zeros(max_length, d_model)
    position = T.arange(max_length).unsqueeze(1)
    div_term = T.exp(T.arange(0, d_model, 2) * - (math.log(10000.0) / d_model))
    pe[:, 0::2] = T.sin(position * div_term)
    pe[:, 1::2] = T.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer("pe", pe)
  
  def forward(self, X: T.Tensor) -> T.Tensor:
    X += T.Variable(self.pe[:, :X.size(1)], requires_grad=False)
    return self.dropout(X)
