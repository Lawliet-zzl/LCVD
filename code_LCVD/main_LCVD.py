"""
Created on Wed Dec 16 2020
@author: xxx
"""
from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import time
import datetime
import math
from tqdm import tqdm

from models import *
from model_func import *
from detectors import *
from data_loader import load_data
from OOD_Metric import evaluate_detection

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument("--decay_epochs", nargs="+", type=int, default=[100, 150], 
	help="decay learning rate by decay_rate at these epochs")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default="resnet", type=str, help='model type (default: resnet)')
parser.add_argument('--dataset', default="CIFAR10", type=str, help='dataset')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=19930815, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int, help='total epochs to run')
parser.add_argument('--decay', default=5e-4, type=float, help='weight decay 5e-4')
parser.add_argument('--precision', default=100000, type=float)
parser.add_argument('--num-classes', default=10, type=int, help='the number of classes (default: 10)')
parser.add_argument('--alg', default='DA', type=str, help='name of algorithm')
parser.add_argument('--save', default=False, action='store_true', help='Save model')
parser.add_argument('--pretrained', default=False, action='store_true', help='load model')
parser.add_argument('--detector', default='baseline', type=str, help='detector')
parser.add_argument('--test', default=False, action='store_true', help='test')
parser.add_argument('--olist', default='A', type=str, help='olist')
parser.add_argument('--pth_path', default='checkpoint/', type=str, help='pth_path')

parser.add_argument('--alpha', default=0, type=float, help='alpha')
parser.add_argument('--beta', default=1, type=float, help='beta 0.5')
parser.add_argument('--order', default=20, type=int, help='order')

parser.add_argument('--dx', default='RM', type=str, help='loss')
parser.add_argument('--dy', default='NE', type=str, help='loss')

args = parser.parse_args()

args.num_classes = 100 if args.dataset == 'imagenet' or args.dataset == 'CIFAR100' else 10
args.batch_size = 64 if args.dataset == 'imagenet' else 128
args.lr = 0.01 if args.dataset == 'imagenet' else 0.1
args.img_size = 224 if args.dataset == 'imagenet' else 32

class NELoss(nn.Module):
	"""docstring for NELoss"""
	def __init__(self):
		super(NELoss, self).__init__()
	def forward(self, outputs):
		loss = (F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1)).sum(dim=1).mean()
		return loss
		
class DALoss(nn.Module):
	"""docstring for DALoss"""
	def __init__(self):
		super(DALoss, self).__init__()
		self.LogSoftmax = nn.LogSoftmax(dim=1)
		self.Softmax = nn.Softmax(dim=1)
		self.KLLoss = nn.KLDivLoss(reduction='batchmean')
		self.CELoss = nn.CrossEntropyLoss()
		self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')
		self.NELoss = NELoss()
		self.eps = 1e-5
	def forward(self, outputs_ID, outputs_OOD, targets_ID, target_list_OOD):
		loss_ID = self.CELoss(outputs_ID, targets_ID)
		res = - torch.log(1 - self.Softmax(outputs_OOD) + self.eps)

		loss_OOD = 0
		if args.dy == 'NL':
			num_OOD = outputs_OOD.size(dim = 0)
			mask = torch.zeros(num_OOD, args.num_classes).cuda()
			for i in range(len(target_list_OOD)):
				for j in range(num_OOD):
					mask[j][target_list_OOD[i][j]] = 1
			loss_OOD = (res * mask).sum() / mask.sum()
		elif args.dy == 'CE':
			loss_OOD = self.CELoss(outputs_OOD, targets_ID)
		elif args.dy == 'TP':
			loss_OOD = self.KLDivLoss(F.log_softmax(outputs_OOD, dim=1), F.softmax(outputs_ID, dim = 1).data / 10)
		elif args.dy == 'NE':
			loss_OOD = self.NELoss(outputs_OOD)
		elif args.dy == 'KL':
			num_OOD = outputs_OOD.size(dim = 0)
			uniform_dist = torch.Tensor(num_OOD, args.num_classes).fill_((1./args.num_classes)).cuda()
			loss_OOD = self.KLDivLoss(F.log_softmax(outputs_OOD, dim=1), uniform_dist)
		elif args.dy == 'SM':
			num_OOD = outputs_OOD.size(dim = 0)
			uniform_dist = torch.Tensor(num_OOD, args.num_classes).fill_((1./args.num_classes)).cuda()
			loss_OOD = 0.1 * self.KLDivLoss(F.log_softmax(outputs_OOD, dim=1), uniform_dist) + 0.9 * self.CELoss(outputs_OOD, targets_ID)

		loss = loss_ID + args.beta*loss_OOD
		return loss_ID, loss_OOD, loss

def stacking_data(inputs, labels, order):
	stacked_data = 0
	label_list = []
	label_list.append(labels)
	for i in range(order - 1):
		index = torch.randperm(inputs.size(0))
		stacked_data = stacked_data + inputs[index, :]
		label_list.append(labels[index])

	if args.alpha == 0:
		stacked_data = (stacked_data + inputs) / order
	else:
		stacked_data = stacked_data / (order - 1)
		stacked_data = (1 - args.alpha) * stacked_data + args.alpha * inputs

	return stacked_data, label_list

def noise_data(inputs, labels):
	num_OOD = inputs.size(0)
	# label_list = []
	# label_list.append(labels)
	inputs_OOD = 0.5 * torch.randn(num_OOD, 3, args.img_size, args.img_size).cuda() + 0.5 * inputs
	return inputs_OOD, labels

def rotate_samples(inputs, labels):
	num_OOD = inputs.size(0)
	label_list = []
	label_list.append(labels)
	inputs_OOD = inputs
	rotations = torch.randint(0, 4, (num_OOD,))
	for i in range(num_OOD):
		inputs_OOD[i] = torch.rot90(inputs_OOD[i], rotations[i] , [1,2])
	return inputs_OOD, label_list

def train_DA(trainloader, net, optimizer, epoch):
	net.train()
	criterion = DALoss()
	train_loss = np.array([0.0,0.0,0.0])
	correct = 0
	total = 0

	total_time = 0.0

	for idx, (inputs, targets) in enumerate(trainloader):

		num_ID = inputs.size(0)
		num_OOD = inputs.size(0)

		inputs_ID, targets_ID = inputs.cuda(), targets.cuda()

		start_time = datetime.datetime.now()
		if args.dx == 'RM':
			inputs_OOD, target_list_OOD = stacking_data(inputs_ID, targets_ID, order = args.order)
		elif args.dx == 'GN':
			inputs_OOD, target_list_OOD = noise_data(inputs_ID, targets_ID)
		elif args.dx == 'RT':
			inputs_OOD, target_list_OOD = rotate_samples(inputs_ID, targets_ID)
		end_time = datetime.datetime.now()
		throughput =  (end_time - start_time) / targets_ID.size(0)
		print(idx, 'throughput', throughput)

		inputs_all = torch.cat((inputs_ID.cpu(), inputs_OOD.cpu()), dim=0).cuda()

		outputs_all = net(inputs_all)
		outputs_ID = outputs_all[0:num_ID,:]
		outputs_OOD = outputs_all[num_ID:num_ID + num_OOD,:]

		loss_ID, loss_OOD, loss = criterion(outputs_ID, outputs_OOD, targets_ID, target_list_OOD)

		train_loss[0] += loss_ID.item()
		train_loss[1] += loss_OOD.item()
		train_loss[2] += loss.item()

		_, predicted = torch.max(outputs_ID.data, 1)
		total += targets_ID.size(0)
		correct += (predicted == targets_ID).sum().item()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	train_loss = train_loss/idx
	train_acc = 100.*correct/total

	return train_loss, train_acc

def detect_OOD(detection_name, net, testloader, OOD_list):
	print("Measuring the out-of-distribution detection performance.")
	mean, std = data_loader.get_known_mean_std(args.dataset)

	with open(detection_name, 'w') as logfile:
		logwriter = csv.writer(logfile, delimiter=',')
		logwriter.writerow(['network','in-dist','out-of-dist', 'AUROC', 'AUPR(IN)', 'AUPR(OUT)', 'FPR(95)', 'Detection'])

	soft_ID = OOD_Metric.get_softmax(net, testloader)

	for dataset_OOD in OOD_list:
		dataloader_OOD = data_loader.load_test_data(mean, std, dataset_OOD)
		soft_OOD = OOD_Metric.get_softmax(net, dataloader_OOD)
		detection_results = OOD_Metric.detect_OOD(soft_ID, soft_OOD, precision=args.precision)
		with open(detection_name, 'a') as logfile:
			logwriter = csv.writer(logfile, delimiter=',')
			logwriter.writerow([args.model, args.dataset, dataset_OOD,
				detection_results[0], 
				detection_results[1], 
				detection_results[2], 
				detection_results[3], 
				detection_results[4]])
	return detection_results

def main():

	OOD_list = getOODlist(args.olist, args.dataset)
	log_name, detection_name, table_name, pth_name = init_setting(args.seed, args.alg, args.dataset, args.model, args.name, args.detector, OOD_list)
	trainloader, testloader = load_data(args.dataset, args.dataset, args.batch_size)
	net = build_model(args.model, args.dataset, args.num_classes)

	name = args.name

	if args.pretrained:
		print("Loading pretrained model...")
		load_pretrained(net, args.pth_path + 'pretrained_' + args.dataset + '_' + args.model + '_0.pth')
		args.lr = args.lr / 100

	criterion =  nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

	print("Training: " + pth_name)
	for epoch in range(0, args.epoch):
		train_loss, train_acc = train_DA(trainloader, net, optimizer, epoch)
		test_loss, test_acc = test(testloader, net, criterion)
		adjust_learning_rate(args.decay_epochs, optimizer, epoch)
		save_result(log_name, epoch, train_loss, train_acc, test_loss, test_acc, optimizer)

	OOD_list = ['CUB200']
	detection_results = detect_OOD(detection_name, net, testloader, OOD_list)
	save_model(pth_name, net)

if __name__ == '__main__':
	main()