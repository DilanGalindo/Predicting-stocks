import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import utils as ut
from torch.utils.data import Dataset, DataLoader




class ConvNet(nn.Module):
	def __init__(self, trg_vocab_size=6, dropout =0.00, device="cuda"):
		super(ConvNet, self).__init__()
		#32 x 29
		#16 16 29
		# 8 8 29
		# 4 4 29
		# 2 2 29
		# 1 1 29
		# 29
		# 6
		self.device = device
		self.dropout = nn.Dropout(p=dropout)
		self.conv1 = nn.Conv2d(29, 25, kernel_size=3, stride=1, padding=(2,2)).to(device)
		# 25, 8, 8
		self.avg1 = nn.AvgPool2d(kernel_size=3, stride=1).to(device)
		# 25,6, 6
		self.layerNorm1 = nn.LayerNorm([25, 6, 6]).to(device)
		self.reLU = nn.ReLU()
		
		self.conv2 = nn.Conv2d(25, 21, kernel_size=3, stride=1, padding= (2,2)).to(device)
		#21, 8, 8
		self.avg2 = nn.AvgPool2d(kernel_size=3, stride=1).to(device)
		#21, 6, 6
		self.layerNorm2 = nn.LayerNorm([21, 6, 6]).to(device)

		self.conv3 = nn.Conv2d(21, 17, kernel_size=3, stride=1, padding=(2,2)).to(device)
		#17, 8, 8
		self.avg3 = nn.AvgPool2d(kernel_size=3, stride=1).to(device)
		#17, 6, 6
		self.layerNorm3 = nn.LayerNorm([17, 6, 6]).to(device)

		self.conv4 = nn.Conv2d(17, 13, kernel_size=3, stride=1).to(device)
		#13, 4, 4
		self.avg4 = nn.AvgPool2d(kernel_size=3, stride=2).to(device)
		#13, 1, 1
		self.layerNorm4 = nn.LayerNorm([13, 1, 1]).to(device).to(device)
		
		self.linear = nn.Linear(13, trg_vocab_size).to(device)
		

	def forward(self, x):
		x = x.to(self.device)
		values = x.transpose(1,2)
		values = values.reshape(values.shape[0],  values.shape[1], 6, 6)
		
		layer1 = self.layerNorm1(self.avg1(self.reLU(self.dropout(self.conv1(values)))))
		
		layer2 = self.layerNorm2(self.avg2(self.reLU(self.dropout(self.conv2(layer1)))))
		
		layer3 = self.layerNorm3(self.avg3(self.reLU(self.dropout(self.conv3(layer2)))))
		
		layer4 = self.layerNorm4(self.avg4(self.reLU(self.dropout(self.conv4(layer3)))))
		out = self.linear(layer4.reshape(layer4.shape[0], -1))
		return out
		
