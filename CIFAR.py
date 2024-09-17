import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Lambda

import numpy as np

num_cores = 36

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
