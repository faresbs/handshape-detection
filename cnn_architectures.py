import torch
import torch.nn as nn
from torchvision import models


class network_vgg16(nn.Module):

	def __init__(self, num_classes):
		super(network_vgg16, self).__init__()
		features = list(models.vgg16().features)
		self.features = nn.ModuleList(features)
		self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes))


	def forward(self, x):
		for feature in self.features:
			x = feature(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)

		return x


class network_inceptionv3(nn.Module):

	def __init__(self, num_classes):
		super(network_inceptionv3, self).__init__()
		features = list(models.inception_v3().features)
		self.features = nn.ModuleList(features)
		self.fc = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes))


	def forward(self, x):
		for feature in self.features:
			x = feature(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x