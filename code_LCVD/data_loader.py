"""
Created on Wed Dec 16 2020
@author: xxx
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset
from PIL import Image
import os
import sys
import os.path
import numpy as np
import math
from numpy.random import randint

class LoadDataset(Dataset):
	"""docstring for LoadDataset"""
	def __init__(self,  root, list_file='train', transform=None, target_transform=None, full_dir=True):
		super(LoadDataset, self).__init__()
		self.root = root
		self.list_file = list_file
		self.transform = transform
		self.target_transform = target_transform
		self.full_dir = full_dir
		self._parse_list()

	def _load_image(self, directory):
		if self.full_dir:
			return Image.open(directory).convert('RGB')
		else:
			return Image.open(os.path.join(self.root, 'data', directory)).convert('RGB')

	def _parse_list(self):
		self.data_list = [LoadRecord(x.strip().split(' ')) for x in open(os.path.join(self.root, self.list_file))]

	def __getitem__(self, index):
		record = self.data_list[index]

		return self.get(record)

	def get(self, record, indices=None):
		img = self._load_image(record.path)

		process_data = self.transform(img)
		if not self.target_transform == None:
			process_label = self.target_transform(record.label)
		else:
			process_label = record.label

		return process_data, process_label

	def __len__(self):
		return len(self.data_list)

class LoadRecord(object):
	"""docstring for LoadRecord"""
	def __init__(self, data):
		super(LoadRecord, self).__init__()
		self._data = data

	@property
	def path(self):
		return self._data[0]

	@property
	def label(self):
		return int(self._data[1])
		
def generate_all_list(root='../data224/CUB_200_2011/'):
	path = root + 'data'
	classname = os.listdir(path)
	classnum = len(classname)
	#random.shuffle(classname)
	for i in range(classnum):
		images = os.listdir(os.path.join(path, classname[i]))
		m = 'w' if i == 0 else 'a'
		with open(os.path.join(root, 'all_list.txt'), m) as f:
			for j in range(len(images)):
				f.write(classname[i] + '/')
				f.write(images[j] + ' ' + str(i))
				f.write('\n')

def getCIFAR10(mean, std, batch_size=128, shuffle=True, train=True, test=True):

	transform = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
		])

	ds = []

	if train:
		print("Loading CIFAR10 training dataset (in-distribution)")
		trainset = datasets.CIFAR10(root='../data_CAM/CIFAR10', train=True, download=True, transform=transform)

		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
		ds.append(trainloader)
	if test:
		print("Loading CIFAR10 testing dataset (in-distribution)")
		testset = datasets.CIFAR10(root='../data_CAM/CIFAR10', train=False, download=True, transform=transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
		ds.append(testloader)

	ds = ds[0] if len(ds) == 1 else ds
	return ds

def getSVHN(mean, std, batch_size=128, shuffle=True, train=True, test=True):

	transform = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
		])

	ds = []

	if train:
		print("Loading SVHN training dataset (in-distribution)")
		trainset = datasets.SVHN(root='../data_CAM/SVHN', train=True, download=True, transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
		ds.append(trainloader)
	if test:
		print("Loading SVHN testing dataset (in-distribution)")
		testset = datasets.SVHN(root='../data_CAM/SVHN', train=False, download=True, transform=transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
		ds.append(testloader)

	ds = ds[0] if len(ds) == 1 else ds
	return ds

def getLSUN(mean, std, version, batch_size=128, shuffle=True):
	transform = transforms.Compose([
	transforms.Resize((32,32)),
	transforms.ToTensor(),
	transforms.Normalize(mean, std)
	])
	if version == 'crop':
		print("Loading LSUN (c) dataset (out-of-distribution)")
		dataset = datasets.ImageFolder('../data_CAM/LSUN_crop',transform=transform)
	elif version == 'resize':
		print("Loading LSUN (r) dataset (out-of-distribution)")
		dataset = datasets.ImageFolder('../data_CAM/LSUN_resize',transform=transform)

	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=9)
	return dataloader

def getTinyImagenet(mean, std, version, batch_size=128, shuffle=True):
	transform = transforms.Compose([
	transforms.Resize((32,32)),
	transforms.ToTensor(),
	transforms.Normalize(mean, std)
	])
	if version == 'crop':
		print("Loading TinyImageNet (c) dataset (out-of-distribution)")
		dataset = datasets.ImageFolder('../data_CAM/TinyImageNet_crop',transform=transform)
	elif version == 'resize':
		print("Loading TinyImageNet (r) dataset (out-of-distribution)")
		dataset = datasets.ImageFolder('../data_CAM/TinyImageNet_resize',transform=transform)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=9)
	return dataloader

def get_known_mean_std(dataset):
	if dataset == 'CIFAR10':
		mean = (0.4914, 0.4822, 0.4465)
		std = (0.2470, 0.2435, 0.2616)
	elif dataset == 'SVHN':
		mean = (0.4377, 0.4438, 0.4728)
		std = (0.1980, 0.2010, 0.1970)
	return mean, std

def load_test_data(mean, std, dataset, batch_size = 100):
	testloader = getAllOOD(root = '../data_LCVD/' + dataset + '/',transform = transform_test, 
			batch_size=100, shuffle=False, glist=False)
	return testloader
	
def getAllOOD(root, transform, batch_size=128, shuffle=False, glist=False):
	if glist:
		generate_all_list(root=root)
	dataset = LoadDataset(root=root, list_file='all_list.txt', transform=transform, full_dir=False)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
	return dataloader

def load_train_ID(mean, std, dataset, batch_size):

	if dataset == 'CIFAR10':
		trainloader = getCIFAR10(mean, std, batch_size=batch_size, shuffle=True, train=True, test=False)
		testloader = getCIFAR10(mean, std, batch_size=100, shuffle=False, train=False, test=True)
	elif dataset == 'SVHN':
		trainloader = getSVHN(mean, std, batch_size=batch_size, shuffle=True, train=True, test=False)
		testloader = getSVHN(mean, std, batch_size=100, shuffle=False, train=False, test=True)
	return trainloader, testloader



