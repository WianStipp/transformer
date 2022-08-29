import torch as T
from torch import nn
import torch.nn.functional as F

N_HEADS = 8
D_MODEL = 512
D_K = 64
D_V = 64

class MultiHeadAttention(nn.Module):
  def __init__(self, n_heads: int, d_model: int, d_k: int, d_v: int) -> None:
    super().__init__()
    self.d_model = d_model
    self.d_k = d_k
    self.d_v = d_v
    self.heads = [Attention() for _ in range(n_heads)]
    self.key_projections = [nn.Linear(d_model, d_k) for _ in range(n_heads)]
    self.query_projections = [nn.Linear(d_model, d_k) for _ in range(n_heads)] 
    self.value_projections = [nn.Linear(d_model, d_v) for _ in range(n_heads)] 
    self.W_O = nn.Linear(n_heads*d_v, d_model)

  def forward(self, Q: T.Tensor, K: T.Tensor, V: T.Tensor, mask: T.Tensor) -> T.Tensor:
    concat_head_outputs = T.concat((head(self.query_projections[i](Q), self.key_projections[i](K), self.value_projections[i](V)) for i, head in enumerate(self.heads)))
    return self.W_O(concat_head_outputs)
    

class Attention(nn.Module):
  """Attention module"""

  def forward(self, Q: T.Tensor, K: T.Tensor, V: T.Tensor) -> T.Tensor:
    """
    Args:
      Q: Query matrix of dims n*d
      K: Key matrix with dims m*d
      V: Value matrix with dims m*v
    """
    d = K.shape[-1]
    attention_scores = F.softmax(T.matmul(Q, K.T), dim=1) * d**(-0.5)
    return T.matmul(attention_scores, V)
