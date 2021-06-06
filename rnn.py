import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
import time
import utils as ut
import tracemalloc
torch.set_default_dtype(torch.float32)

class PositionalEncoding(nn.Module):
	def __init__(self, d_model=36, dropout=0.1, max_len=5000, width = 29, device="cuda"):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		max_len=1

		#1#N
		pe = torch.zeros(max_len, d_model, width).to(device)
		#N X 1
		#1X32
		for pos in range(0, width):
			for i in range(0, d_model, 2):
				iT = float(i)
				div_term = math.exp(iT * (-math.log(10000.0) / d_model))
				pe[:,i, pos] = math.sin(pos*div_term)
				pe[:,i+1,pos] = math.cos(pos*div_term)

		pe = pe.to(device)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:,:, :x.size(2)]

		return self.dropout(x)


class SelfAttention(nn.Module):
	"""docstring for SelfAttention"""
	#Heads how many parts we split it
	def __init__(self, embed_size =36.0, heads = 4.0, padding =8, num_layers=9, device="cuda"):
		super(SelfAttention, self).__init__()
		self.embed_size = embed_size
		self.device= device
		
		self.heads = heads
		self.head_dim = embed_size // heads

		assert (self.head_dim * heads == embed_size), "Embed size needs ot be div by heads"

		self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
		self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
		self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

		self.fc_out = nn.Linear(heads*self.head_dim, embed_size)
		self.padding = (num_layers*-1 + padding*2)*2
		self.mask = (torch.triu(torch.ones(self.heads, 9, 9).to(device), diagonal=-padding).to(device) == 1)		
		
		self.mask = self.mask.float().masked_fill(self.mask == 1, float('-1e20')).masked_fill(self.mask == 0, float(0.0))
		self.mask = self.mask.to(self.device)[None,:,:,:]
		
						
		
	def forward(self, values, keys, query, mask):
		#N how many example we are sending at the same time
		Nquery = query.shape[0]
		Nvalues = values.shape[0]
		Nkey = keys.shape[0]
		value_len, key_len, query_len = values.shape[2],  keys.shape[2], query.shape[2]

		#Split embediing into self.heads pieces
		values = values.reshape(Nvalues, value_len, self.heads, self.head_dim)
		keys = keys.reshape(Nkey, key_len, self.heads, self.head_dim)
		query = query.reshape(Nquery, query_len, self.heads, self.head_dim)


		values = self.values(values)
		keys = self.keys(keys)
		query = self.queries(query)


		#energy shape (N, heads, query_len, key_len)
		#query shape (N, query_len, heads, head_dim)
		#keys shape (N, key_len, heads, head_dim)

		energy = torch.einsum("nqhd,nkhd->nhqk", [query, keys]).to(self.device)

		if mask is not None:
			energy = energy.masked_fill(self.mask == 0, float("-1e20"))
			

		attention = torch.softmax(energy / (self.embed_size ** (1.0/2.0)), dim=3).to(self.device)

		#attention shape (N, heads, query_len, key_len)
		#values shape (N, value_len, heads, head_dim)
		#out shape (N, query_len, heads, head_dim)

		#after einsum: concatentate the informaiton
		out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
			Nquery, query_len, self.heads*self.head_dim
			).to(self.device)

		out = self.fc_out(out)

		return out.transpose(1,2)

class TransformerBlock(nn.Module):
	def __init__(self, embed_size, heads, dropout, forward_expansion, padding=8, num_layers=9, device="cuda"  ):
		super(TransformerBlock, self).__init__()
		self.attention = SelfAttention(embed_size, heads, padding, num_layers, device)
		self.embed_size = embed_size
		#print("TransformerBlock padding", padding)
		

		self.feed_forward = nn.Sequential(
			nn.Linear(embed_size, forward_expansion*embed_size),
			nn.ReLU(),
			nn.Linear(forward_expansion*embed_size, embed_size)
			)
		self.dropout = nn.Dropout(dropout)

	def forward(self, value, key, query, mask):
		attention = self.attention(value, key, query, mask)

		self.norm1 = nn.LayerNorm([ self.embed_size, attention.shape[2]])
		self.norm2 = nn.LayerNorm([ self.embed_size, attention.shape[2]])

		#Add & Norm
		firstLayer = self.dropout(self.norm1(attention + query))
		forward = self.feed_forward(firstLayer.transpose(1,2)).transpose(1,2)

		out = self.dropout(self.norm2(forward+firstLayer))
		return out

class Encoder(nn.Module):
	def __init__(self, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
		super(Encoder, self).__init__()
		#Max length positional embeding = 29
		#Vocal size = 29
		self.embed_size = embed_size
		self.device = device

		self.layers = nn.ModuleList(
			[
			TransformerBlock(
				embed_size,
				heads,
				dropout=dropout,
				forward_expansion = forward_expansion,
				device = device
				)
			for _ in range(num_layers)
			]
			)
		self.dropout = nn.Dropout(dropout)
		self.dropout_value = dropout
		self.position = PositionalEncoding(self.embed_size, dropout, device=device)




	def forward(self, x, mask):
		
		out = self.position(x)

		for layer in self.layers:
			out = layer(out, out, out, mask)
			#print("Encoder Positional shape", out.shape)

		return out


class DecoderBlock(nn.Module):
	"""docstring for DecoderBlock"""
	def __init__(self, embed_size, heads, forward_expansion, dropout, device, padding, num_layers):
		super(DecoderBlock, self).__init__()
		self.attention = SelfAttention(embed_size, heads, padding, num_layers, device)
		self.norm = nn.LayerNorm([embed_size,9])
		self.transformer_block = TransformerBlock(
			embed_size, heads, dropout, forward_expansion, padding, num_layers, device
			)
		self.padding = padding
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, value, key, scr_mask, trg_mask):
		
		attention = self.attention(x,x,x, trg_mask)
		query = self.dropout(self.norm(attention + x))
		

		out = self.transformer_block(value, key, query, None)

		return out


class Decoder(nn.Module):
	"""docstring for Decoder"""
	def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length, padding ):
		super(Decoder, self).__init__()
		self.device = device
		self.position_embedding = nn.Embedding(max_length, embed_size)
		

		self.layers = nn.ModuleList(
			[DecoderBlock(embed_size, heads, forward_expansion, dropout, device, padding-i, num_layers)
			for i in range(num_layers)]
			)

		self.fc_out = nn.Linear(embed_size*9, 36)
		self.fc_out_final = nn.Linear(36, trg_vocab_size)
		self.dropout = nn.Dropout(dropout)
		self.embed_size = embed_size
		self.dropout_value = dropout
		self.max_length = max_length
		self.position = PositionalEncoding(self.embed_size, dropout, device=device)
		self.trg_vocab_size = trg_vocab_size



	def forward(self, x, enc_out, src_mask, trg_mask):
		x = self.position(x).to(self.device)
		
		for layer in self.layers:
			x = layer(x, enc_out, enc_out, src_mask, trg_mask)
			
		out = self.fc_out(x.reshape(x.shape[0], -1))
		out = self.fc_out_final(out)
		#print("out shape", out.shape)

		return out


class Transformer(nn.Module):
	"""docstring for Transformer"""
	def __init__(self, trg_vocab_size, embed_size=36, num_layers=9, forward_expansion=4, heads = 4, dropout =0.01, device="cuda", max_length = 15):
		super(Transformer, self).__init__()
		
		self.position = PositionalEncoding(embed_size, dropout, device=device)

		self.encoder = Encoder( embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
		self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length, num_layers-1)
		self.heads = heads
		self.embed_size = embed_size
		self.device = device
		self.position_embedding = nn.Embedding(max_length, embed_size).to(device)


	def make_trg_mask(self, trg, heads):
		sz = trg.shape[1]
		sz2 = trg.shape[2]
		mask = (torch.triu(torch.ones(heads, 9, sz2).to(self.device), diagonal=-9).to(self.device) == 1)
	

		mask = mask.float().masked_fill(mask == 1, float('-1e20')).masked_fill(mask == 0, float(0.0))
		return mask.to(self.device)[None,:,:,:]

	def forward(self, src, trg):
		trg = trg.to(self.device)
		src = src.to(self.device)
		trg_res = torch.zeros(trg.shape[0], self.embed_size, trg.shape[1]).to(self.device)
		for idx, _ in enumerate(trg):
			trg_res[idx,:,:] = self.position_embedding(trg[idx]).transpose(0,1)

		del trg
		trg_res = trg_res.to(self.device)


		trg_mask = self.make_trg_mask(trg_res, self.heads)
		enc_src = self.encoder(src, None)
		out = self.decoder(trg_res, enc_src, None, trg_mask)
		return out

