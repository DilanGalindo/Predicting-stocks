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

		# pe[:,:, 0::2] = torch.sin(position * div_term)
		# pe[:,:, 1::2] = torch.cos(position * div_term)
		# pe = pe.unsqueeze(0).transpose(0, 1).to(device)
		self.register_buffer('pe', pe)

	def forward(self, x):
		#print(self.pe.shape, x.shape, "Positional", x.dtype)
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
		
		#print("Attention padding", self.padding)
						
		
	def forward(self, values, keys, query, mask):
		#N how many example we are sending at the same time
		Nquery = query.shape[0]
		Nvalues = values.shape[0]
		Nkey = keys.shape[0]
		value_len, key_len, query_len = values.shape[2],  keys.shape[2], query.shape[2]
		#print("SelfAttention Begin values size", values.shape,"keys", keys.shape, "query",query.shape, "heads",self.head_dim)

		#Split embediing into self.heads pieces
		values = values.reshape(Nvalues, value_len, self.heads, self.head_dim)
		keys = keys.reshape(Nkey, key_len, self.heads, self.head_dim)
		query = query.reshape(Nquery, query_len, self.heads, self.head_dim)

		#print("SelfAttention After values size", values.shape, "keys", keys.shape, "query",query.shape)

		values = self.values(values)
		keys = self.keys(keys)
		query = self.queries(query)

		#print("SelfAttention End values size", values.shape, "keys",keys.shape, "query",query.shape)

		#energy shape (N, heads, query_len, key_len)
		#query shape (N, query_len, heads, head_dim)
		#keys shape (N, key_len, heads, head_dim)

		energy = torch.einsum("nqhd,nkhd->nhqk", [query, keys]).to(self.device)
		#print("Energy energy shape", energy.shape)

		if mask is not None:
			#print(self.mask.shape, energy.shape, query.shape, keys.shape)
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
		#print("SelfAttention Final out size", out.shape, "transpose", out.transpose(1,2).shape)
		#print()
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
		#print("TransformerBlock attention", attention.shape, "Query shape", query.shape, "LayerNorm",self.embed_size, attention.shape[2] )

		self.norm1 = nn.LayerNorm([ self.embed_size, attention.shape[2]])
		self.norm2 = nn.LayerNorm([ self.embed_size, attention.shape[2]])

		#Add & Norm
		firstLayer = self.dropout(self.norm1(attention + query))
		#print("TransformerBlock Firstlayer", firstLayer.shape, "tranpose", firstLayer.transpose(1,2).shape)
		forward = self.feed_forward(firstLayer.transpose(1,2)).transpose(1,2)
		#print("TransformerBlock forward", forward.shape)

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
		#print("Encoder shape", x.shape)
		
		out = self.position(x)
		#print("Encoder shape", x.shape, "Positional shape", out.shape)

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
		#print("padding", padding)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, value, key, scr_mask, trg_mask):
		
		##print("Decoder Before", x.shape, trg_mask.shape)
		attention = self.attention(x,x,x, trg_mask)
		#print("Decoder After", self.padding)
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
		#print("Decoder enc shape", enc_out.shape, x.shape, self.embed_size)
		#snapshot = tracemalloc.take_snapshot()
		#top_stats = snapshot.statistics('lineno')
		##print("[ Top 10 ]")
		#for stat in top_stats[:]:
			##print(stat)
		
		#print()
		#print()
		#print()
		x = self.position(x).to(self.device)
		
		for layer in self.layers:
			x = layer(x, enc_out, enc_out, src_mask, trg_mask)
			
		#print("out shape", x.reshape(x.shape[0], -1).shape, self.embed_size, self.trg_vocab_size, self.embed_size*9)

		out = self.fc_out(x.reshape(x.shape[0], -1))
		out = self.fc_out_final(out)
		#print("out shape", out.shape)

		return out


class Transformer(nn.Module):
	"""docstring for Transformer"""
	def __init__(self, trg_vocab_size, embed_size=36, num_layers=9, forward_expansion=4, heads = 4, dropout =0.01, device="cuda", max_length = 15):
		super(Transformer, self).__init__()
		# strart end + - dot numbers 0 -9 5 numbers
		#max_length 15
		#target vocalSize 9
		
		self.position = PositionalEncoding(embed_size, dropout, device=device)

		self.encoder = Encoder( embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
		self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length, num_layers-1)
		self.heads = heads
		self.embed_size = embed_size
		self.device = device
		self.position_embedding = nn.Embedding(max_length, embed_size).to(device)


	def make_trg_mask(self, trg, heads):
		#print(trg.shape)
		sz = trg.shape[1]
		sz2 = trg.shape[2]
		mask = (torch.triu(torch.ones(heads, 9, sz2).to(self.device), diagonal=-9).to(self.device) == 1)
	

		mask = mask.float().masked_fill(mask == 1, float('-1e20')).masked_fill(mask == 0, float(0.0))
		#mask = mask.transpose(0,1)
		return mask.to(self.device)[None,:,:,:]

	def forward(self, src, trg):
		trg = trg.to(self.device)
		src = src.to(self.device)
		#print("Shape src", src.shape, src.dtype)
		#print("Shape target before embeding", trg.shape)
		trg_res = torch.zeros(trg.shape[0], self.embed_size, trg.shape[1]).to(self.device)
		for idx, _ in enumerate(trg):
			#print("Shape target index embeding ", idx,trg[idx].shape, trg[idx].dtype)
			trg_res[idx,:,:] = self.position_embedding(trg[idx]).transpose(0,1)

		del trg
		trg_res = trg_res.to(self.device)
		#print("Shape target after embeding", trg_res.shape)


		trg_mask = self.make_trg_mask(trg_res, self.heads)
		#print("Mask", trg_mask.shape)
		enc_src = self.encoder(src, None)
		out = self.decoder(trg_res, enc_src, None, trg_mask)
		return out

"""
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

criterion = nn.L1Loss().to(device)
lr = 0.001 # learning rate
model = Transformer( trg_vocab_size=1, embed_size=36, num_layers=9, forward_expansion=4, heads = 4, dropout =0.00, device=device, max_length = 15).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

dictionaryOutput = {"s": 0, "e":1, "+": 2, "-": 3, ".": 4, "0": 5, "1": 6, "2": 7, "3": 8, "4": 9, "5": 10, "6": 11, "7": 12, "8": 13, "9": 14}


dataset = ut.quarterDataset("ADI" )
#138
batch_size = 10
dataLoader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0)


torch.cuda.empty_cache()
dataiter = iter(dataLoader)
data = dataiter.next()
features, labels1, labels2, labels3, labels4, labels5 = data
model.train()
num_epoch = 10



res = torch.zeros(labels1.shape[0], 9, dtype=torch.int).to(device)
for idx, _ in enumerate(labels1):
	pos_neg = "+" if (int(labels1[idx])>=0) else ""

	output = "s"+pos_neg+"{0:03d}".format(int(labels1[idx])) + "{:.2f}".format(float(labels1[idx]))[-3:]+"e"
	#output = "s+000.00e"
	outputArray = []
	for i in output:
		outputArray.append(dictionaryOutput[i])
	outputArray = torch.IntTensor(outputArray).to(device)
	res[idx] = outputArray
#tracemalloc.start()
t = torch.cuda.get_device_properties(0).total_memory
r = torch.cuda.memory_reserved(0) 
a = torch.cuda.memory_allocated(0)
f = r-a  # free inside reserved
print("memory", t,r,a,f)

compareOutputArray = torch.empty(batch_size, 1).to(device)
torch.autograd.set_detect_anomaly(True)
for epoch in range(num_epoch):
	model.train()
	
	for idx, Y in enumerate(labels1):
	 compareOutputArray[idx,:] = Y
		
	out = model(features, res)
			
	loss = criterion(out, compareOutputArray)
	
	optimizer.zero_grad()
	loss.backward()
	
	optimizer.step()
	scheduler.step(loss)

model.eval()
length = float(len(labels1))

for idx, _ in enumerate(labels1):
	pos_neg = "+" if (int(labels1[idx])>=0) else ""

	output = "s+000.00e"
	outputArray = []
	for i in output:
		outputArray.append(dictionaryOutput[i])
	outputArray = torch.IntTensor(outputArray)
	res[idx] = outputArray


out = model(features, res)
loss = criterion(out, compareOutputArray)
valid = 0
for idx, (Y, Z) in enumerate(zip(labels1, out)):

	print(Y, Z)
	valid = valid +1 if((Y + (Y*.1)>= Z) and (Y-(Y*.1))<= Z) else valid
	
print(valid/length, valid)

"""