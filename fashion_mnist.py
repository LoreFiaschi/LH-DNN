import torch

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Lambda


num_cores = 36

class FashionMnist():
	def __init__(self, batch_size):

		self.class_levels = 3
		self.num_c1 = 2
		self.num_c2 = 6
		self.num_c3 = 10
		self.batch_size = batch_size
		self.training_size = 60000
		self.test_size = 10000

		#--- coarse 1 classes ---
		self.labels_c1 = ('clothes', 'goods')
		#--- coarse 2 classes ---
		self.labels_c2 = ('tops', 'bottoms', 'dresses', 'outers', 'accessories', 'shoes')
		#--- fine classes ---
		self.labels_c3 = ('t-shirt','trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', "ankle boot")
			   
		transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5), (0.5))])
		#transform = transforms.ToTensor()

		coarser = Lambda(lambda y: torch.tensor([self.c3_to_c1(y), self.c3_to_c2(y), int(y)]))

		trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform, target_transform = coarser)
		
		self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_cores)
	
		testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform, target_transform = coarser)

		self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_cores)
	

	def c3_to_c1(self, y):
		match y:
			case 0: # t-shirt
				return 0
			case 1: # trouser
				return 0
			case 2: # pullover
				return 0
			case 3: # dress
				return 0
			case 4: # coat
				return 0
			case 5: # sandal
				return 1
			case 6: # shirt
				return 0
			case 7: # sneaker
				return 1
			case 8: # bag
				return 1
			case 9: # ankle boot
				return 1

	def c3_to_c2(self, y):
		match y:
			case 0: # t-shirt
				return 0
			case 1: # trouser
				return 1
			case 2: # pullover
				return 0
			case 3: # dress
				return 2
			case 4: # coat
				return 3
			case 5: # sandal
				return 5
			case 6: # shirt
				return 0
			case 7: # sneaker
				return 5
			case 8: # bag
				return 4
			case 9: # ankle boot
				return 5

	def c2_to_c1(self, y):
		if y < 4:
			return 0

		return 1
#		match y:
#			case 0: # tops
#				return 0
#			case 1: # bottoms
#				return 0
#			case 2: # dresses
#				return 0
#			case 3: # outers
#				return 0
#			case 4: # accessories
#				return 1
#			case 5: # shoes
#				return 1
		
	
	def c2_reinforce(self, c1_logits, c2_reinforcer):
		num_rows = c1_logits.size(0)
		c2_reinforcer[:num_rows,0] = c2_reinforcer[:num_rows,1] = c2_reinforcer[:num_rows,2] = c2_reinforcer[:num_rows,3] = c1_logits[:,0]
		c2_reinforcer[:num_rows,4] = c2_reinforcer[:num_rows,5] = c1_logits[:,1]
		
		return c2_reinforcer[:num_rows,:]
		
	
	def c3_reinforce(self, c2_logits, c3_reinforcer):
		num_rows = c2_logits.size(0)
		c3_reinforcer[:num_rows,0] = c3_reinforcer[:num_rows,2] = c3_reinforcer[:num_rows,6] = c2_logits[:,0]
		c3_reinforcer[:num_rows,1] = c2_logits[:,1]
		c3_reinforcer[:num_rows,3] = c2_logits[:,2]
		c3_reinforcer[:num_rows,4] = c2_logits[:,3]
		c3_reinforcer[:num_rows,8] = c2_logits[:,4]
		c3_reinforcer[:num_rows,5] = c3_reinforcer[:num_rows,7] = c3_reinforcer[:num_rows,9] = c2_logits[:,5]
		
		return c3_reinforcer[:num_rows,:]
	

	def __str__(self):
		return "FashionMnist"
