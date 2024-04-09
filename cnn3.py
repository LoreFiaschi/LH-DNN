import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
#from torchviz import make_dot
from torch.linalg import vector_norm as vnorm
from torch.linalg import solve as solve_matrix_system

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Lambda

from abc import ABC
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 


num_cores = 36
torch.set_num_interop_threads(num_cores) # Inter-op parallelism
torch.set_num_threads(num_cores) # Intra-op parallelism
device = "cuda:0"
#device = "cpu"

class CIFAR10():

	def __init__(self, batch_size):

		self.class_levels = 3
		self.num_c1 = 2
		self.num_c2 = 7
		self.num_c3 = 10
		self.batch_size = batch_size
		self.training_size = 50000

		#--- coarse 1 classes ---
		self.labels_c1 = ('transport', 'animal')
		#--- coarse 2 classes ---
		self.labels_c2 = ('sky', 'water', 'road', 'bird', 'reptile', 'pet', 'medium')
		#--- fine classes ---
		self.labels_c3 = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
			   
		transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

		coarser = Lambda(lambda y: torch.tensor([self.c3_to_c1(y), self.c3_to_c2(y), int(y)]))

		self.batch_size = 128

		trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform, target_transform = coarser)

		self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_cores)

		testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform, target_transform = coarser)

		self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_cores)


	def c3_to_c1(self, y):
		if y < 2 or y > 7:
			return 0
		return 1

	def c3_to_c2(self, y):
		match y:
			case 0:
				return 0
			case 1:
				return 2
			case 2:
				return 3
			case 3:
				return 5
			case 4:
				return 6
			case 5:
				return 5
			case 6:
				return 4
			case 7:
				return 6
			case 8:
				return 1
			case _:
				return 2

	def c2_to_c1(self, y):
		if y < 3:
			return 0
		return 1
		
	
	def c2_reinforce(self, c1_logits, c2_reinforcer):
		num_rows = c1_logits.size(0)
		c2_reinforcer[:num_rows,0] = c2_reinforcer[:num_rows,1] = c2_reinforcer[:num_rows,2] = c1_logits[:,0]
		c2_reinforcer[:num_rows,3] = c2_reinforcer[:num_rows,4] = c2_reinforcer[:num_rows,5] = c2_reinforcer[:num_rows,6] = c1_logits[:,1]
		
		return c2_reinforcer[:num_rows,:]
		
	
	def c3_reinforce(self, c2_logits, c3_reinforcer):
		num_rows = c2_logits.size(0)
		c3_reinforcer[:num_rows,0] = c2_logits[:,0]
		c3_reinforcer[:num_rows,8] = c2_logits[:,1]
		c3_reinforcer[:num_rows,1] = c3_reinforcer[:num_rows,9] = c2_logits[:,2]
		c3_reinforcer[:num_rows,2] = c2_logits[:,3]
		c3_reinforcer[:num_rows,6] = c2_logits[:,4]
		c3_reinforcer[:num_rows,3] = c3_reinforcer[:num_rows,5] = c2_logits[:,5]
		c3_reinforcer[:num_rows,4] = c3_reinforcer[:num_rows,7] = c2_logits[:,6]
		
		return c3_reinforcer[:num_rows,:]
		
	
	def __str__(self):
		return "CIFAR10"


class CIFAR100():

	def __init__(self, batch_size):

		self.class_levels = 3
		self.num_c1 = 8
		self.num_c2 = 20
		self.num_c3 = 100
		self.batch_size = batch_size
		self.training_size = 50000
			   
		transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

		coarser = Lambda(lambda y: torch.tensor([self.c3_to_c1(y), self.c3_to_c2(y), int(y)]))

		self.batch_size = 128

		trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform, target_transform = coarser)

		self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_cores)

		testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform, target_transform = coarser)

		self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_cores)
		
		self.labels_c3 = [
			"apple",
			"aquarium_fish",
			"baby",
			"bear",
			"beaver",
			"bed",
			"bee",
			"beetle",
			"bicycle",
			"bottle",
			"bowl",
			"boy",
			"bridge",
			"bus",
			"butterfly",
			"camel",
			"can",
			"castle",
			"caterpillar",
			"cattle",
			"chair",
			"chimpanzee",
			"clock",
			"cloud",
			"cockroach",
			"couch",
			"crab",
			"crocodile",
			"cup",
			"dinosaur",
			"dolphin",
			"elephant",
			"flatfish",
			"forest",
			"fox",
			"girl",
			"hamster",
			"house",
			"kangaroo",
			"keyboard",
			"lamp",
			"lawn_mower",
			"leopard",
			"lion",
			"lizard",
			"lobster",
			"man",
			"maple",
			"motorcycle",
			"mountain",
			"mouse",
			"mushroom",
			"oak",
			"orange",
			"orchid",
			"otter",
			"palm",
			"pear",
			"pickup_truck",
			"pine",
			"plain",
			"plate",
			"poppy",
			"porcupine",
			"possum",
			"rabbit",
			"raccoon",
			"ray",
			"road",
			"rocket",
			"rose",
			"sea",
			"seal",
			"shark",
			"shrew",
			"skunk",
			"skyscraper",
			"snail",
			"snake",
			"spider",
			"squirrel",
			"streetcar",
			"sunflower",
			"sweet_pepper",
			"table",
			"tank",
			"telephone",
			"television",
			"tiger",
			"tractor",
			"train",
			"trout",
			"tulip",
			"turtle",
			"wardrobe",
			"whale",
			"willow",
			"wolf",
			"woman",
			"worm"
		]

	
		self.labels_c2 = [
			"aquatic_mammals",
			"fish",
			"flowers",
			"food_containers",
			"fruit_and_vegetables",
			"household_electrical_devices",
			"household_furniture",
			"insects",
			"large_carnivores",
			"large_man-made_outdoor_things",
			"large_natural_outdoor_scenes",
			"large_omnivores_and_herbivores",
			"medium-sized_mammals",
			"non-insect_invertebrates",
			"people",
			"reptiles",
			"small_mammals",
			"trees",
			"vehicles_1",
			"vehicles_2"
		]
		
		self.labels_c1 = [
			"water_animals",
			"flora",
			"tools",
			"tiny_animals",
			"animals",
			"views",
			"people",
			"vehicles"
		]
	
	def c3_to_c1(self, y):
		return self.c2_to_c1( self.c3_to_c2(y) )

	def c3_to_c2(self, y):
	
		if self.labels_c3[y] == "beaver" or self.labels_c3[y] == "dolphin" or self.labels_c3[y] == "otter" or self.labels_c3[y] == "seal" or self.labels_c3[y] == "whale":
			return 0
			
		if self.labels_c3[y] == "aquarium_fish" or self.labels_c3[y] == "flatfish" or self.labels_c3[y] == "ray" or self.labels_c3[y] == "shark" or self.labels_c3[y] == "trout":
			return 1
			
		if self.labels_c3[y] == "orchid" or self.labels_c3[y] == "poppy" or self.labels_c3[y] == "rose" or self.labels_c3[y] == "sunflower" or self.labels_c3[y] == "tulip":
			return 2
			
		if self.labels_c3[y] == "bottle" or self.labels_c3[y] == "bowl" or self.labels_c3[y] == "can" or self.labels_c3[y] == "cup" or self.labels_c3[y] == "plate":
			return 3
	
		if self.labels_c3[y] == "apple" or self.labels_c3[y] == "mushroom" or self.labels_c3[y] == "orange" or self.labels_c3[y] == "pear" or self.labels_c3[y] == "sweet_pepper":
			return 4
			
		if self.labels_c3[y] == "clock" or self.labels_c3[y] == "keyboard" or self.labels_c3[y] == "lamp" or self.labels_c3[y] == "telephone" or self.labels_c3[y] == "television":
			return 5
			
		if self.labels_c3[y] == "bed" or self.labels_c3[y] == "chair" or self.labels_c3[y] == "couch" or self.labels_c3[y] == "table" or self.labels_c3[y] == "wardrobe":
			return 6
			
		if self.labels_c3[y] == "bee" or self.labels_c3[y] == "beetle" or self.labels_c3[y] == "butterfly" or self.labels_c3[y] == "caterpillar" or self.labels_c3[y] == "cockroach":
			return 7
			
		if self.labels_c3[y] == "bear" or self.labels_c3[y] == "leopard" or self.labels_c3[y] == "lion" or self.labels_c3[y] == "tiger" or self.labels_c3[y] == "wolf":
			return 8
			
		if self.labels_c3[y] == "bridge" or self.labels_c3[y] == "castle" or self.labels_c3[y] == "house" or self.labels_c3[y] == "road" or self.labels_c3[y] == "skyscraper":
			return 9
			
		if self.labels_c3[y] == "cloud" or self.labels_c3[y] == "forest" or self.labels_c3[y] == "mountain" or self.labels_c3[y] == "plain" or self.labels_c3[y] == "sea":
			return 10
			
		if self.labels_c3[y] == "camel" or self.labels_c3[y] == "cattle" or self.labels_c3[y] == "chimpanzee" or self.labels_c3[y] == "elephant" or self.labels_c3[y] == "kangaroo":
			return 11
			
		if self.labels_c3[y] == "fox" or self.labels_c3[y] == "porcupine" or self.labels_c3[y] == "possum" or self.labels_c3[y] == "raccoon" or self.labels_c3[y] == "skunk":
			return 12
			
		if self.labels_c3[y] == "crab" or self.labels_c3[y] == "lobster" or self.labels_c3[y] == "snail" or self.labels_c3[y] == "spider" or self.labels_c3[y] == "worm":
			return 13
			
		if self.labels_c3[y] == "baby" or self.labels_c3[y] == "boy" or self.labels_c3[y] == "girl" or self.labels_c3[y] == "man" or self.labels_c3[y] == "woman":
			return 14
			
		if self.labels_c3[y] == "crocodile" or self.labels_c3[y] == "dinosaur" or self.labels_c3[y] == "lizard" or self.labels_c3[y] == "snake" or self.labels_c3[y] == "turtle":
			return 15
			
		if self.labels_c3[y] == "hamster" or self.labels_c3[y] == "mouse" or self.labels_c3[y] == "rabbit" or self.labels_c3[y] == "shrew" or self.labels_c3[y] == "squirrel":
			return 16
			
		if self.labels_c3[y] == "maple" or self.labels_c3[y] == "oak" or self.labels_c3[y] == "palm" or self.labels_c3[y] == "pine" or self.labels_c3[y] == "willow":
			return 17
			
		if self.labels_c3[y] == "bicycle" or self.labels_c3[y] == "bus" or self.labels_c3[y] == "motorcycle" or self.labels_c3[y] == "pickup_truck" or self.labels_c3[y] == "train":
			return 18
			
		if self.labels_c3[y] == "lawn_mower" or self.labels_c3[y] == "rocket" or self.labels_c3[y] == "streetcar" or self.labels_c3[y] == "tank" or self.labels_c3[y] == "tractor":
			return 19
			
		raise Exception(f'Label {y} does not exist.')

	def c2_to_c1(self, y):
	
		if y == 0 or y == 1:
			return 0
			
		if y == 2 or y == 4 or y == 17:
			return 1
			
		if y == 3 or y == 5 or y == 6:
			return 2
		
		if y == 7 or y == 13:
			return 3
		
		if y == 8 or y == 11 or y == 12 or y == 15 or y == 16:
			return 4
			
		if y == 9 or y == 10:
			return 5
			
		if y == 14:
			return 6
			
		return 7
		
	
	def c1_to_c2(self, y):
		match y:
			case 0:
				return [0,1]
			case 1:
				return [2,4,17]
			case 2:
				return [3,5,6]
			case 3:
				return [7, 13]
			case 4:
				return [8,11,12,15,16]
			case 5:
				return [9,10]
			case 6:
				return [14]
			case _:
				return [18, 19]
				
	def c2_to_c3(self, y):
		match y:
			case 0:
				return [4, 30, 55, 72, 95]
			case 1:
				return [1, 32, 67, 73, 91]
			case 2:
				return [54, 62, 70, 82, 92]
			case 3:
				return [9, 10, 16, 28, 61]
			case 4:
				return [0, 51, 53, 57, 83]
			case 5:
				return [22, 39, 40, 86, 87]
			case 6:
				return [5, 20, 25, 84, 94]
			case 7:
				return [6, 7, 14, 18, 24]
			case 8:
				return [3, 42, 43, 88, 97]
			case 9:
				return [12, 17, 37, 68, 76]
			case 10:
				return [23, 33, 49, 60, 71]
			case 11:
				return [15, 19, 21, 31, 38]
			case 12:
				return [34, 63, 64, 66, 75]
			case 13:
				return [26, 45, 77, 79, 99]
			case 14:
				return [2, 11, 35, 46, 98]
			case 15:
				return [27, 29, 44, 78, 93]
			case 16:
				return [36, 50, 65, 74, 80]
			case 17:
				return [47, 52, 56, 59, 96]
			case 18:
				return [8, 13, 48, 58, 90]
			case 19:
				return [41, 69, 81, 85, 89]
				
							
	def print_tree(self):
		for i in np.arange(self.num_c1):
			idx_2 = self.c1_to_c2(i)
			idx_3 = self.c2_to_c3(idx_2[0])
			print(f'{self.labels_c1[i]:15s} -> {self.labels_c2[idx_2[0]]:25s} -> {self.labels_c3[idx_3[0]]:15s}')
			
			for label_3 in idx_3[1:]:
				print(f'{"":<15}	{"":<25}	{self.labels_c3[label_3]:15s}')
				
			for j, label_2 in enumerate(idx_2[1:]):
				idx_3 = self.c2_to_c3(idx_2[j+1])
				print(f'{"":<15}	{self.labels_c2[label_2]:25s} -> {self.labels_c3[idx_3[0]]:15s}')
				
				for label_3 in idx_3[1:]:
					print(f'{"":<15}	{"":<25}	{self.labels_c3[label_3]:15s}')
					
					
	def c2_reinforce(self, c1_logits, c2_reinforcer):
		num_rows = c1_logits.size(0)
		c2_reinforcer[:num_rows,0] = c2_reinforcer[:num_rows,1] = c1_logits[:,0]
		c2_reinforcer[:num_rows,2] = c2_reinforcer[:num_rows,4] = c2_reinforcer[:num_rows,17] = c1_logits[:,1]
		c2_reinforcer[:num_rows,3] = c2_reinforcer[:num_rows,5] = c2_reinforcer[:num_rows,6] = c1_logits[:,2]
		c2_reinforcer[:num_rows,7] = c2_reinforcer[:num_rows,13] = c1_logits[:,3]
		c2_reinforcer[:num_rows,8] = c2_reinforcer[:num_rows,11] = c2_reinforcer[:num_rows,12] = c2_reinforcer[:num_rows,15] = c2_reinforcer[:num_rows,16] = c1_logits[:,4]
		c2_reinforcer[:num_rows,9] = c2_reinforcer[:num_rows,10] = c1_logits[:,5]
		c2_reinforcer[:num_rows,14] = c1_logits[:,6] 
		c2_reinforcer[:num_rows,18] = c2_reinforcer[:num_rows,19] = c1_logits[:,7]
		
		return c2_reinforcer[:num_rows,:]
		
	def c3_reinforce(self, c2_logits, c3_reinforcer):
		num_rows = c2_logits.size(0)
		c3_reinforcer[:num_rows,4] = c3_reinforcer[:num_rows,30] = c3_reinforcer[:num_rows,55] = c3_reinforcer[:num_rows,72] = c3_reinforcer[:num_rows,95] = c2_logits[:,0]
		c3_reinforcer[:num_rows,1] = c3_reinforcer[:num_rows,32] = c3_reinforcer[:num_rows,67] = c3_reinforcer[:num_rows,73] = c3_reinforcer[:num_rows,91] = c2_logits[:,1]
		c3_reinforcer[:num_rows,54] = c3_reinforcer[:num_rows,62] = c3_reinforcer[:num_rows,70] = c3_reinforcer[:num_rows,82] = c3_reinforcer[:num_rows,92] = c2_logits[:,2]
		c3_reinforcer[:num_rows,9] = c3_reinforcer[:num_rows,10] = c3_reinforcer[:num_rows,16] = c3_reinforcer[:num_rows,28] = c3_reinforcer[:num_rows,61] = c2_logits[:,3]
		c3_reinforcer[:num_rows,0] = c3_reinforcer[:num_rows,51] = c3_reinforcer[:num_rows,53] = c3_reinforcer[:num_rows,57] = c3_reinforcer[:num_rows,83] = c2_logits[:,4]
		c3_reinforcer[:num_rows,22] = c3_reinforcer[:num_rows,39] = c3_reinforcer[:num_rows,40] = c3_reinforcer[:num_rows,86] = c3_reinforcer[:num_rows,87] = c2_logits[:,5]
		c3_reinforcer[:num_rows,5] = c3_reinforcer[:num_rows,20] = c3_reinforcer[:num_rows,25] = c3_reinforcer[:num_rows,84] = c3_reinforcer[:num_rows,94] = c2_logits[:,6]
		c3_reinforcer[:num_rows,6] = c3_reinforcer[:num_rows,7] = c3_reinforcer[:num_rows,14] = c3_reinforcer[:num_rows,18] = c3_reinforcer[:num_rows,24] = c2_logits[:,7]
		c3_reinforcer[:num_rows,3] = c3_reinforcer[:num_rows,42] = c3_reinforcer[:num_rows,43] = c3_reinforcer[:num_rows,88] = c3_reinforcer[:num_rows,97] = c2_logits[:,8]
		c3_reinforcer[:num_rows,12] = c3_reinforcer[:num_rows,17] = c3_reinforcer[:num_rows,37] = c3_reinforcer[:num_rows,68] = c3_reinforcer[:num_rows,76] = c2_logits[:,9]
		c3_reinforcer[:num_rows,23] = c3_reinforcer[:num_rows,33] = c3_reinforcer[:num_rows,49] = c3_reinforcer[:num_rows,60] = c3_reinforcer[:num_rows,71] = c2_logits[:,10]
		c3_reinforcer[:num_rows,15] = c3_reinforcer[:num_rows,19] = c3_reinforcer[:num_rows,21] = c3_reinforcer[:num_rows,31] = c3_reinforcer[:num_rows,38] = c2_logits[:,11]
		c3_reinforcer[:num_rows,34] = c3_reinforcer[:num_rows,63] = c3_reinforcer[:num_rows,64] = c3_reinforcer[:num_rows,66] = c3_reinforcer[:num_rows,75] = c2_logits[:,12]
		c3_reinforcer[:num_rows,26] = c3_reinforcer[:num_rows,45] = c3_reinforcer[:num_rows,77] = c3_reinforcer[:num_rows,79] = c3_reinforcer[:num_rows,99] = c2_logits[:,13]
		c3_reinforcer[:num_rows,2] = c3_reinforcer[:num_rows,11] = c3_reinforcer[:num_rows,35] = c3_reinforcer[:num_rows,46] = c3_reinforcer[:num_rows,98] = c2_logits[:,14]
		c3_reinforcer[:num_rows,27] = c3_reinforcer[:num_rows,29] = c3_reinforcer[:num_rows,44] = c3_reinforcer[:num_rows,78] = c3_reinforcer[:num_rows,93] = c2_logits[:,15]
		c3_reinforcer[:num_rows,36] = c3_reinforcer[:num_rows,50] = c3_reinforcer[:num_rows,65] = c3_reinforcer[:num_rows,74] = c3_reinforcer[:num_rows,80] = c2_logits[:,16]
		c3_reinforcer[:num_rows,47] = c3_reinforcer[:num_rows,52] = c3_reinforcer[:num_rows,56] = c3_reinforcer[:num_rows,59] = c3_reinforcer[:num_rows,96] = c2_logits[:,17]
		c3_reinforcer[:num_rows,8] = c3_reinforcer[:num_rows,13] = c3_reinforcer[:num_rows,48] = c3_reinforcer[:num_rows,58] = c3_reinforcer[:num_rows,90] = c2_logits[:,18]
		c3_reinforcer[:num_rows,41] = c3_reinforcer[:num_rows,69] = c3_reinforcer[:num_rows,81] = c3_reinforcer[:num_rows,85] = c3_reinforcer[:num_rows,89] = c2_logits[:,19]
		
		return c3_reinforcer[:num_rows,:]


	def __str__(self):
		return "CIFAR100"
		
		

class CNN3(ABC, nn.Module):
	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, every_print = 512, switch_points = None, custom_training = False, threshold = 0.0, reduction = 'mean', only_thresholded = False):
		
		super().__init__()
		self.dataset = dataset
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.nesterov = nesterov
		self.activation = F.relu
		self.epochs = epochs
		self.switch_points = switch_points
		self.custom_training = custom_training
		self.every_print = every_print - 1 # assumed power of 2, -1 to make the mask
		self.track_size = int( self.dataset.training_size / self.dataset.batch_size / every_print ) 
		self.threshold = threshold
		self.predict_and_learn = self.predict_and_learn_naive if threshold == 0.0 else (self.predict_and_learn_only_thresholded if only_thresholded else self.predict_and_learn_thresholded)
		self.reduction = reduction
		self.only_thresholded = only_thresholded
		
		self.c2_reinforcer = torch.empty((self.dataset.batch_size, self.dataset.num_c2), device = device)
		self.c3_reinforcer = torch.empty((self.dataset.batch_size, self.dataset.num_c3), device = device)

		self.v = torch.tensor(0., device = device)
		
		if switch_points is not None and len(switch_points) != len(learning_rate)-1:
			raise ValueError("switch_points must have the one element less than learning_rate ")

	
	def forward(self, x):
		pass


	# Assumption: W is column full rank. 
	def project(self, z): # https://math.stackexchange.com/questions/4021915/projection-orthogonal-to-two-vectors

		W1 = self.layerb17.weight.clone().detach()
		W2 = self.layerb27.weight.clone().detach()
		ort2 = torch.empty_like(z)
		ort3 = torch.empty_like(z)
		
		zz = z.clone().detach()

		mask = torch.heaviside(zz, self.v)
		Rk = torch.einsum('ij, jh -> ijh', mask, self.I)
		W1k = W1.matmul(Rk)
		W2k_ = W2.matmul(Rk)
		W2k = torch.cat((W1k, W2k_), dim = 1)

		ort2 = self.compute_orthogonal(z, W1k, self.I_o1)
		ort3 = self.compute_orthogonal(z, W2k, self.I_o2)

		self.ort2 = ort2.clone().detach()
		self.ort3 = ort3.clone().detach()

		prj2 = zz - self.ort2
		prj3 = zz - self.ort3

		return ort2, ort3, prj2, prj3

	def compute_orthogonal(self, z, W, I_o, eps = 1e-8):
		WWT = torch.matmul(W, W.mT)
		P = solve_matrix_system(WWT + torch.randn_like(WWT, device = device) * eps, I_o) # Broadcasting
		P = torch.matmul(P, W)
		P = self.I - torch.matmul(W.mT, P) # Broadcasting
		
		return torch.matmul(z.unsqueeze(1), P).squeeze()
		
	
	def predict_and_learn_naive(self, batch, labels):
		self.optimizer.zero_grad()
		predict = self(batch)
		loss_f = self.criterion(predict[0], labels[:,0]) #+ 1e-5 * sum(sum(abs(self.layerb17.weight)))
		loss_i1 = self.criterion(predict[1], labels[:,1]) #+ 1e-5 * sum(sum(abs(self.layerb27.weight)))
		loss_i2 = self.criterion(predict[2], labels[:,2])
		
		loss_f.backward(retain_graph=True)
		loss_i1.backward(retain_graph=True)
		loss_i2.backward()

		self.optimizer.step()

		return torch.tensor([loss_f, loss_i1, loss_i2]).clone().detach(), torch.tensor([torch.heaviside(self.ort2-1e-7, self.v).sum(dim=1).mean(), torch.heaviside(self.ort3-1e-7, self.v).sum(dim=1).mean()]).clone().detach()
		

	def vect_to_scalar_loss(self, loss_vect):
		return torch.mean(loss_vect[torch.where(loss_vect >= self.threshold)])

	
	def predict_and_learn_thresholded(self, batch, labels):
		self.optimizer.zero_grad()
		predict = self(batch)
		loss_f_vect = self.criterion(predict[0], labels[:,0]) 
		loss_i1_vect = self.criterion(predict[1], labels[:,1])
		loss_i2_vect = self.criterion(predict[2], labels[:,2])

		loss_f_m = torch.mean(loss_f_vect).clone().detach()
		loss_i1_m = torch.mean(loss_i1_vect).clone().detach()
		loss_i2_m = torch.mean(loss_i2_vect).clone().detach()

		back = False

		if loss_f_m >= self.threshold:
			loss_f = self.vect_to_scalar_loss(loss_f_vect)
			loss_f.backward(retain_graph=True)
			back = True
			
		if loss_i1_m >= self.threshold:
			loss_i1 = self.vect_to_scalar_loss(loss_i1_vect)
			loss_i1.backward(retain_graph=True)
			back = True
			
		if loss_i2_m >= self.threshold:
			loss_i2 = self.vect_to_scalar_loss(loss_i2_vect)
			loss_i2.backward()
			back = True

		if back:
			self.optimizer.step()
		
		return torch.tensor([loss_f, loss_i1, loss_i2]).clone().detach(), torch.tensor([torch.heaviside(self.ort2-1e-7, self.v).sum(dim=1).mean(), torch.heaviside(self.ort3-1e-7, self.v).sum(dim=1).mean()]).clone().detach()
		
		
	def predict_and_learn_only_thresholded(self, batch, labels):
		self.optimizer.zero_grad()
		predict = self(batch)
		loss_f_vect = self.criterion(predict[0], labels[:,0]) 
		loss_i1_vect = self.criterion(predict[1], labels[:,1])
		loss_i2_vect = self.criterion(predict[2], labels[:,2])

		loss_f = self.vect_to_scalar_loss(loss_f_vect)
		loss_i1 = self.vect_to_scalar_loss(loss_i1_vect)
		loss_i2 = self.vect_to_scalar_loss(loss_i2_vect)

		back = False

		if loss_f.isnan() == False:
			loss_f.backward(retain_graph=True)
			back = True
			
		if loss_i1_m.isnan() == False:
			loss_i1.backward(retain_graph=True)
			back = True
			
		if loss_i2_m.isnan() == False:
			loss_i2.backward()
			back = True

		if back:
			self.optimizer.step()
		
		return torch.tensor([loss_f, loss_i1, loss_i2]).clone().detach(), torch.tensor([torch.heaviside(self.ort2-1e-7, self.v).sum(dim=1).mean(), torch.heaviside(self.ort3-1e-7, self.v).sum(dim=1).mean()]).clone().detach()
	

	def training_loop_body(self):			
		for batch, labels in self.dataset.trainloader:
			batch = batch.to(device)
			labels = labels.to(device)
			self.predict_and_learn(batch, labels)

			
	def training_loop_body_track(self):
		running_loss = torch.zeros(self.dataset.class_levels)
		running_l0 = torch.zeros(self.dataset.class_levels-1)
		
		iter = 1
			
		for batch, labels in self.dataset.trainloader:
			batch = batch.to(device)
			labels = labels.to(device)
			loss, l0 = self.predict_and_learn(batch, labels)

			running_loss += (loss - running_loss) / iter
			running_l0 += (l0 - running_l0) / iter
			
			if iter & self.every_print == 0:
				self.loss_track[self.num_push, :] = running_loss
				self.accuracy_track[self.num_push, :] = self.test(mode = "train")
				self.l0_track[self.num_push, :] = running_l0
				self.num_push += 1
				running_loss = torch.zeros(self.dataset.class_levels)
				running_l0 = torch.zeros(self.dataset.class_levels-1)
				iter = 1

			iter +=1

	
	def train_model(self, track = False, filename = ""):
		self.I = torch.eye(self.layerb_mid.out_features, device = device)
		self.I_o1 = torch.eye(self.layerb17.out_features, device = device)
		self.I_o2 = torch.eye(self.layerb17.out_features + self.layerb27.out_features, device = device)
		self.train()
		training_f = self.training_loop_body
		
		if track:
			self.loss_track = torch.zeros(self.epochs * self.track_size, self.dataset.class_levels)
			self.accuracy_track = torch.zeros_like(self.loss_track)
			self.l0_track = torch.zeros(self.epochs * self.track_size, self.dataset.class_levels-1)
			self.num_push = 0
			training_f = self.training_loop_body_track

		if self.custom_training:
			self.custom_training_f(track, filename)
			
		elif self.switch_points is None:
			for epoch in tqdm(self.epochs, desc="Training: "):
				training_f()
				
		else:
			prev_switch_point = 0
			for idx, switch_point in enumerate(self.switch_points):
				for epoch in tqdm(np.arange(prev_switch_point, switch_point), desc="Training: "):
					training_f()
				self.optimizer.param_groups[0]['lr'] = self.learning_rate[idx]
				prev_switch_point  = switch_point
				
			for epoch in tqdm(np.arange(prev_switch_point, self.epochs), desc="Training: "):
				training_f()

		if track:
			self.plot_training_loss(filename + "_train_loss.pdf")
			self.plot_test_accuracy(filename + "_test_accuracy_.pdf")
			self.plot_l0(filename + "_l0.pdf")

		
	def custom_training_f(self, track = False, filename = ""):
		pass


	def initialize_memory(self, mode):
		self.correct_c1_pred = torch.zeros(self.dataset.num_c1)
		self.total_c1_pred = torch.zeros_like(self.correct_c1_pred)
		
		self.correct_c2_pred = torch.zeros(self.dataset.num_c2)
		self.total_c2_pred = torch.zeros_like(self.correct_c2_pred)
		
		self.correct_c3_pred = torch.zeros(self.dataset.num_c3)
		self.total_c3_pred = torch.zeros_like(self.correct_c3_pred)
		
		if mode != "train":
		
			self.correct_c1_vs_c2_pred = torch.zeros(self.dataset.num_c1)
			self.total_c1_vs_c2_pred = torch.zeros_like(self.correct_c1_vs_c2_pred)
			
			self.correct_c2_vs_c3_pred = torch.zeros(self.dataset.num_c2)
			self.total_c2_vs_c3_pred = torch.zeros_like(self.correct_c2_vs_c3_pred)

			self.correct_c1_vs_c3_pred = torch.zeros(self.dataset.num_c1)
			self.total_c1_vs_c3_pred = torch.zeros_like(self.correct_c1_vs_c3_pred)

	
	def collect_test_performance(self, mode):
		with torch.no_grad():
			for images, labels in self.dataset.testloader:
				images = images.to(device)
				labels = labels.to(device)
				predictions = self(images)
				predicted = torch.zeros(predictions[0].size(0), self.dataset.class_levels, dtype=torch.long)
				_, predicted[:,0] = torch.max(predictions[0], 1)
				_, predicted[:,1] = torch.max(predictions[1], 1)
				_, predicted[:,2] = torch.max(predictions[2], 1)

				for i in np.arange(predictions[0].size(0)):
					if labels[i,0] == predicted[i,0]:
						self.correct_c1_pred[labels[i,0]] += 1
						
					if labels[i,1] == predicted[i,1]:
						self.correct_c2_pred[labels[i,1]] += 1

					if labels[i,2] == predicted[i,2]:
						self.correct_c3_pred[labels[i,2]] += 1
						
					self.total_c1_pred[labels[i,0]] += 1
					self.total_c2_pred[labels[i,1]] += 1
					self.total_c3_pred[labels[i,2]] += 1
					
					if mode != "train":
						if predicted[i,1] == self.dataset.c3_to_c2(predicted[i,2]):
							self.correct_c2_vs_c3_pred[predicted[i,1]] +=1
							
						if predicted[i,0] == self.dataset.c3_to_c1(predicted[i,2]):
							self.correct_c1_vs_c3_pred[predicted[i,0]] += 1
							
						if predicted[i,0] == self.dataset.c2_to_c1(predicted[i,1]):
							self.correct_c1_vs_c2_pred[predicted[i,0]] += 1
							
						self.total_c1_vs_c3_pred[predicted[i,0]] += 1
						self.total_c1_vs_c2_pred[predicted[i,0]] += 1
						self.total_c2_vs_c3_pred[predicted[i,1]] += 1


	def test_results_to_text(self):
		str_bot = ""
		str = ""
		
		# accuracy for each class
		for i in np.arange(self.dataset.num_c1):
			accuracy_c1 = 100 * float(self.correct_c1_pred[i]) / self.total_c1_pred[i]
			str += f'Accuracy for class {self.dataset.labels_c1[i]:5s}: {accuracy_c1:.2f} %'
			str += '\n'
			
		str += '\n'
		
		for i in np.arange(self.dataset.num_c2):
			accuracy_c2 = 100 * float(self.correct_c2_pred[i]) / self.total_c2_pred[i]
			str += f'Accuracy for class {self.dataset.labels_c2[i]:5s}: {accuracy_c2:.2f} %'
			str += '\n'
			
		str += '\n'
		
		for i in np.arange(self.dataset.num_c3):
			accuracy_c3 = 100 * float(self.correct_c3_pred[i]) / self.total_c3_pred[i]
			str += f'Accuracy for class {self.dataset.labels_c3[i]:5s}: {accuracy_c3:.2f} %'
			str += '\n'
			
		# accuracy for the whole dataset
		str += '\n'

		str += f'Accuracy on c1: {(100 * self.correct_c1_pred.sum() / self.total_c1_pred.sum()):.2f} %'
		str += '\n'

		str_bot += f'Accuracy on c1: {(100 * self.correct_c1_pred.sum() / self.total_c1_pred.sum()):.2f} %'
		str_bot += '\n'

		str += f'Accuracy on c2: {(100 * self.correct_c2_pred.sum() / self.total_c2_pred.sum()):.2f} %'
		str += '\n'

		str_bot += f'Accuracy on c2: {(100 * self.correct_c2_pred.sum() / self.total_c2_pred.sum()):.2f} %'
		str_bot += '\n'

		str += f'Accuracy on c3: {(100 * self.correct_c3_pred.sum() / self.total_c3_pred.sum()):.2f} %'
		str += '\n'

		str_bot += f'Accuracy on c3: {(100 * self.correct_c3_pred.sum() / self.total_c3_pred.sum()):.2f} %'
		str_bot += '\n'

		str += '\n'
		str_bot += '\n'
		
		 
		str += f'Cross-accuracy c1 vs c2: {100 * float(self.correct_c1_vs_c2_pred.sum()) / self.total_c1_vs_c2_pred.sum():.2f} %'
		str += '\n'
		str += f'Cross-accuracy c2 vs c3: {100 * float(self.correct_c2_vs_c3_pred.sum()) / self.total_c2_vs_c3_pred.sum():.2f} %'
		str += '\n'
		str += f'Cross-accuracy c1 vs c3: {100 * float(self.correct_c1_vs_c3_pred.sum()) / self.total_c1_vs_c3_pred.sum():.2f} %'
		str += '\n\n'
 
		str_bot += f'Cross-accuracy c1 vs c2: {100 * float(self.correct_c1_vs_c2_pred.sum()) / self.total_c1_vs_c2_pred.sum():.2f} %'
		str_bot += '\n'
		str_bot += f'Cross-accuracy c2 vs c3: {100 * float(self.correct_c2_vs_c3_pred.sum()) / self.total_c2_vs_c3_pred.sum():.2f} %'
		str_bot += '\n'
		str_bot += f'Cross-accuracy c1 vs c3: {100 * float(self.correct_c1_vs_c3_pred.sum()) / self.total_c1_vs_c3_pred.sum():.2f} %'
		str_bot += '\n'

		# cross classes accuracy (tree)
		for i in np.arange(self.dataset.num_c1):
			accuracy_c1_c2 = 100 * float(self.correct_c1_vs_c2_pred[i]) / self.total_c1_vs_c2_pred[i]
			str += f'Cross-accuracy {self.dataset.labels_c1[i]:9s} vs c2: {accuracy_c1_c2:.2f} %'
			str += '\n'
			
		str += '\n'
		
		for i in np.arange(self.dataset.num_c2):
			accuracy_c2_c3 = 100 * float(self.correct_c2_vs_c3_pred[i]) / self.total_c2_vs_c3_pred[i]
			str += f'Cross-accuracy {self.dataset.labels_c2[i]:7s} vs c3: {accuracy_c2_c3:.2f} %'
			str += '\n'
			
		str += '\n'
		
		for i in np.arange(self.dataset.num_c1):
			accuracy_c1_c3 = 100 * float(self.correct_c1_vs_c3_pred[i]) / self.total_c1_vs_c3_pred[i]
			str += f'Cross-accuracy {self.dataset.labels_c1[i]:9s} vs c3: {accuracy_c1_c3:.2f} %'
			str += '\n'
			

		return str, str_bot
		

	def barplot(self, x, accuracy, labels, title):
		plt.bar(x, accuracy, tick_label = labels)
		plt.xlabel("Classes")
		plt.ylabel("Accuracy")
		plt.title(title)
		plt.show();

	
	def plot_test_results(self):
		# accuracy for each class
		accuracy_c1 = torch.empty(self.dataset.num_c1)
		for i in np.arange(self.dataset.num_c1):
			accuracy_c1[i] = float(self.correct_c1_pred[i]) / self.total_c1_pred[i]
		self.barplot(np.arange(self.dataset.num_c1), accuracy_c1, self.dataset.labels_c1, "Accuracy on the first level")

		accuracy_c2 = torch.empty(self.dataset.num_c2 + 1)
		for i in np.arange(self.dataset.num_c2):
			accuracy_c2[i] = float(self.correct_c2_pred[i]) / self.total_c2_pred[i]
		accuracy_c2[self.dataset.num_c2] = self.correct_c2_pred.sum() / self.total_c2_pred.sum()
		self.barplot(np.arange(self.dataset.num_c2 + 1), accuracy_c2, (*self.dataset.labels_c2, 'overall'), "Accuracy on the second level")

		accuracy_c3 = torch.empty(self.dataset.num_c3 + 1)
		for i in np.arange(self.dataset.num_c3):
			accuracy_c3[i] = float(self.correct_c3_pred[i]) / self.total_c3_pred[i]
		accuracy_c3[self.dataset.num_c3] = self.correct_c3_pred.sum() / self.total_c3_pred.sum()
		self.barplot(np.arange(self.dataset.num_c3 + 1), accuracy_c3, (*self.dataset.labels_c3, 'overall'), "Accuracy on the third level")

	
	def test(self, mode = "print", filename = None):
		self.eval()
		self.initialize_memory(mode)

		self.collect_test_performance(mode)

		match mode:
			case "plot":
				self.plot_test_results()

			case "print":
				msg, msg_bot = self.test_results_to_text()
				print(msg)
				return msg_bot

			case "write":
				msg, msg_bot = self.test_results_to_text()
				with open(filename+"_test_performance.txt", 'w') as f:
					f.write(msg)
				return msg_bot

			case "train":
				accuracy_c1 = self.correct_c1_pred.sum() / self.total_c1_pred.sum()
				accuracy_c2 = self.correct_c2_pred.sum() / self.total_c2_pred.sum()
				accuracy_c3 = self.correct_c3_pred.sum() / self.total_c3_pred.sum()

				self.train()

				return torch.tensor([accuracy_c1, accuracy_c2, accuracy_c3])
				
			case _:
				raise AttributeError("Test mode not available")
		
	
	def plot_training_loss(self, filename):
		plt.figure(figsize=(12, 6))
		plt.plot(np.linspace(1, self.epochs, self.loss_track.size(0)), self.loss_track[:, 0].numpy(), label = "First level")
		plt.plot(np.linspace(1, self.epochs, self.loss_track.size(0)), self.loss_track[:, 1].numpy(), label = "Second level")
		plt.plot(np.linspace(1, self.epochs, self.loss_track.size(0)), self.loss_track[:, 2].numpy(), label = "Third level")
		plt.title("Training loss")
		plt.xlabel("Epochs")
		plt.ylabel("Error")
		plt.xticks(np.linspace(1, self.epochs, self.epochs)[0::2])
		plt.legend()
		plt.savefig(filename, bbox_inches='tight')
		plt.close()

	
	def plot_test_accuracy(self, filename):
		plt.figure(figsize=(12, 6))
		plt.plot(np.linspace(1, self.epochs, self.accuracy_track.size(0)), self.accuracy_track[:, 0].numpy(), label = "First level")
		plt.plot(np.linspace(1, self.epochs, self.accuracy_track.size(0)), self.accuracy_track[:, 1].numpy(), label = "Second level")
		plt.plot(np.linspace(1, self.epochs, self.accuracy_track.size(0)), self.accuracy_track[:, 2].numpy(), label = "Third level")
		plt.title("Test accuracy")
		plt.xlabel("Epochs")
		plt.ylabel("Accuracy")
		plt.xticks(np.linspace(1, self.epochs, self.epochs)[0::2])
		plt.legend()
		plt.savefig(filename, bbox_inches='tight')
		plt.close()
		
			
	def plot_l0(self, filename):
		color2 = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
		color3 = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
		plt.figure(figsize=(12, 6))
		plt.plot(np.linspace(1, self.epochs, self.accuracy_track.size(0)), self.l0_track[:, 0].numpy(), color = color2, label = "Second level")
		plt.plot(np.linspace(1, self.epochs, self.accuracy_track.size(0)), self.l0_track[:, 1].numpy(), color = color3, label = "Third level")
		plt.title("Orthogonal L0 norm")
		plt.xlabel("Epochs")
		plt.ylabel("Norm")
		plt.xticks(np.linspace(1, self.epochs, self.epochs)[0::2])
		plt.legend()
		plt.savefig(filename, bbox_inches='tight')
		plt.close()
		
	
	def save_model(self, path):
		torch.save(self.state_dict(), path+".pt")

	
	def load_model(self, path):
		self.load_state_dict(torch.load(path+".pt"))
		self.eval()

	
	def num_param(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	
	def write_configuration(self, filename, additional_info = "No"):
		msg = ""
		msg += "Epochs: " + str(self.epochs) + "\n\n"
		msg += "LR: " 
		for lr in self.learning_rate:
			msg += str(lr) + " "
		msg += "\n\n"
		msg += "Switch point: "
		for sp in self.switch_points:
			msg += str(sp) + " "
		msg += "\n\n"
		msg += "Number of params: " + str(self.num_param()) + "\n\n"
		msg += "Loss threshold: " + str(self.threshold) + "\n\n"
		msg += "Only thresholded: " + str(self.only_thresholded) + "\n\n"
		msg += "Reduction: " + self.reduction + "\n\n"
		msg += "Additional info: " + additional_info
		
		with open(filename+"_configuration.txt", 'w') as f:
			f.write(msg)
			

	def c2_mask(self, c1_logits):
		c1_labels = torch.argmax(c1_logits, dim = 1)
		mask = torch.ones((c1_logits.size(0), self.dataset.num_c2))
		for i, label in enumerate(c1_labels):
			idx = self.dataset.c1_to_c2(label)
			mask[i, idx] = 2
		
		return mask
		

	def c3_mask(self, c2_logits):
		c2_labels = torch.argmax(c2_logits, dim = 1)
		mask = torch.ones((c2_logits.size(0), self.dataset.num_c3))
		for i, label in enumerate(c2_labels):
			idx = self.dataset.c2_to_c3(label)
			mask[i, idx] = 2
		
		return mask


	def c2_reinforce(self, c1_logits):
		return self.dataset.c2_reinforce(c1_logits, self.c2_reinforcer)
		

	def c3_reinforce(self, c2_logits):
		return self.dataset.c3_reinforce(c2_logits, self.c3_reinforcer)
