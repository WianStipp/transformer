# 1 / 5 / 2022 
# Building blocks of a transformer

import copy
import torch
import torch.nn as nn
from torch.nn import functional as F

class multi_head_attention(nn.Module):

	def __init__(self,
				 n_heads : "int",
				 embedding_len : "int" 
				 ):
		
		super(multi_head_attention, self).__init__()
		assert embedding_len%n_heads==0, "input_size must be possible to split among heads"
		
		# dimension sizes
		self.n_heads = n_heads
		self.embedding_len = embedding_len
		self.head_len = embedding_len // n_heads

		# linear layers with weights for attention and outputs
		self.W_query, self.W_keys, self.W_values = [ 
			nn.Linear(self.head_len,self.head_len,bias=False) 
			for _ in range(3) 
		]
		self.W_out = nn.Linear(embedding_len,embedding_len)

	def forward(self,query,key,value,mask=None):

		n_batches = query.size(0)

		# split inputs in to the heads
		h_query, h_key, h_value = [
			layer( tensor.reshape( n_batches, tensor.size(1), self.n_heads, self.head_len ) )
			for tensor, layer in zip( [query,key,value], [self.W_query, self.W_keys, self.W_values] )
		]

		scores = torch.einsum("nqhd,nkhd->nhqk",[h_query,h_key])
		# query_shape :  (n_batches, query_len, n_heads, dims_per_head)
		# key_shape :    (n_batches, key_len, n_heads, dims_per_head)
		# scores_shape : (n_batches, n_heads, query_len, key_len)

		# apply mask to set 0 weight on masked values
		if mask is not None: scores.masked_fill(mask==0,-1e20)

		normed_scores = scores / self.embedding_len**0.5

		attention_weights = F.softmax( normed_scores, dim = -1 )

		headwise_attention = torch.einsum( "nhqk,nvhd->nqhd",[attention_weights, h_value])
		# attention_weights_shape : (n_batches, n_heads, query_len, key_len)
		# value_shape :             (n_batches, value_len, n_heads, dims_per_head)
		# headwise_attention :      (n_batches, query_len, n_heads, dims_per_head)

		# concatenate the output from the heads and apply linear output layer to them
		attended_values = headwise_attention.reshape(n_batches,query.size(1),self.embedding_len)
		out = self.W_out( attended_values )
		return out

class transformer_cell(nn.Module):

	def __init__(self,n_heads,embedding_len, dropout_proba = 0, forward_expansion = 1):
		
		super(transformer_cell,self).__init__()
		
		self.attention = multi_head_attention(n_heads,embedding_len)

		# 
		self.norm_dropout_1, self.norm_dropout_2= [ 
			nn.Sequential( 
				nn.LayerNorm(embedding_len), 
				nn.Dropout(dropout_proba)
			)
			for _ in range(2) 
		]

		# fully connected feed forward network 
		self.feed_forward = nn.Sequential(
			nn.Linear(embedding_len, forward_expansion*embedding_len),
			nn.ReLU(),
			nn.Linear(forward_expansion*embedding_len, embedding_len),
		)

	def forward(self,query,key,value,mask = None):
		attended_values = self.attention(query,key,value,mask)
		dropout_normed_values = self.norm_dropout_1(attended_values + query)
		forward_passed = self.feed_forward(dropout_normed_values)
		out = self.norm_dropout_2(forward_passed + dropout_normed_values)
		return out

class encoder(nn.Module):

	def __init__(self, 
			  n_heads,
			  embedding_len,
			  forward_expansion,
			  dropout_proba,
			  n_layers,
			  vocab_len, 
			  max_length,
			  device,
			  ):
		
		super(encoder,self).__init__()

		self.embedding_len = embedding_len
		self.device = device
		self.word_embedding = nn.Embedding(vocab_len,embedding_len)
		self.position_embedding = nn.Embedding(max_length,embedding_len)

		self.layers = nn.ModuleList(
			[ 
				transformer_cell(n_heads,embedding_len,dropout_proba,forward_expansion)
				for _ in range(n_layers)
			]	
		)

		self.dropout = nn.Dropout(dropout_proba)

	def forward(self,x,mask):

		n_batches, seq_len = x.shape
		positions = torch.arange(0,seq_len).expand(n_batches,seq_len).to(self.device)
		embedded_token = self.word_embedding(x) + self.position_embedding(positions)
		out = self.dropout(embedded_token)

		for layer in self.layers:
			out = layer(out,out,out,mask)

		return out

class decoder_block(nn.Module):

	def __init__(self, 
			  n_heads,
			  embedding_len,
			  forward_expansion,
			  dropout_proba,
			  device,
			  ):
		
		super(decoder_block,self).__init__()

		self.attention = multi_head_attention(n_heads,embedding_len)
		self.norm = nn.LayerNorm(embedding_len)
		self.transformer = transformer_cell(
			n_heads, embedding_len, dropout_proba, forward_expansion
		)
		self.dropout = nn.Dropout(dropout_proba)

	def forward(self,x,value,key,source_mask,target_mask):

		attention = self.attention(x,x,x,target_mask)
		query = self.dropout(self.norm(attention+x))
		out = self.transformer(query,key,value,source_mask)
		return out

class decoder(nn.Module):

	def __init__(self,
			  n_heads,
			  embedding_len,
			  forward_expansion,
			  dropout_proba,
			  device,		
			  n_layers,
			  target_vocab_len,
			  max_length
			  ):
		super(decoder,self).__init__()
		self.device = device
		self.word_embedding = nn.Embedding(target_vocab_len, embedding_len)
		self.position_embedding = nn.Embedding(max_length,embedding_len)

		self.layers = nn.ModuleList(
			[ 
				decoder_block(n_heads,embedding_len, forward_expansion,dropout_proba,device)
				for _ in range(n_layers)
			]	
		)

		self.linear = nn.Linear(embedding_len,target_vocab_len)
		self.dropout = nn.Dropout( dropout_proba )

	def forward(self, x, encoder_output, source_mask, target_mask):

		n_batches, sequence_length = x.shape
		positions = torch.arange(0,sequence_length).expand(n_batches,sequence_length).to(self.device)
		embedded_token = self.word_embedding(x) + self.position_embedding(positions)

		out = self.dropout(embedded_token)
		for layer in self.layers:
			out = layer(embedded_token,encoder_output,encoder_output,source_mask,target_mask)

		return out

class transformer_model(nn.Module):

	def __init__(self,
			  source_vocab_len,
			  target_vocab_len,
			  source_pad_index,
			  target_pad_index,
			  embedding_len = 256,
			  n_layers = 6,
			  forward_expansion = 4,
			  n_heads = 8,
			  dropout_proba = 0,
			  device = "cpu",
			  max_length = 100
			  ):
		super(transformer_model,self).__init__()

		self.encoder = encoder(
			n_heads,
			embedding_len,
			forward_expansion,
			dropout_proba,
			n_layers,
			source_vocab_len,
			max_length,
			device
		)

		self.decoder = decoder(
			n_heads,
			embedding_len,
			forward_expansion,
			dropout_proba,
			device,
			n_layers, 
			target_vocab_len,
			max_length
		)

		self.source_pad_index=source_pad_index
		self.target_pad_index=target_pad_index
		self.device = device

	def make_source_mask(self,source):

		source_mask = (source != self.source_pad_index ).unsqueeze(1).unsqueeze(2)
		return source_mask.to(self.device)

	def make_target_mask(self,target):
		n_batches, target_length = target.shape
		target_mask = torch.tril(torch.ones((target_length,target_length))).expand(
			n_batches, 1, target_length,target_length
		)
		return target_mask.to(self.device)	
	
	def forward(self,source,target):
		source_mask = self.make_source_mask(source)
		target_mask = self.make_target_mask(source)
		encoded_source = self.encoder(source,source_mask)
		out = self.decoder(target, encoded_source,source_mask,target_mask)
		return out

sentence = torch.randint(0,10,(5,20),dtype=torch.long)
a = transformer_model(100,100,0,0,256,2,2,4)
out = a.forward(sentence,sentence)
print("end")