import torch as T
from torch import nn
import torch.nn.functional as F

class Feedforward(nn.Module):
  def __init__(self, d_model: int, dropout: float=0.1) -> None:
    super().__init__()
    self.d_model = d_model
    self.dropout = nn.Dropout(dropout)
    self.W1 = nn.Linear(d_model, d_model)
    self.W2 = nn.Linear(d_model, d_model)
    
  def forward(self, X: T.Tensor) -> T.Tensor:
    return self.W2(self.dropout(F.relu(self.W1(X))))
