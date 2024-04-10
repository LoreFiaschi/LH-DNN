import cnn3

from cnn3 import torch, optim
from cnn3 import CNN3
from cnn3 import nn
from cnn3 import np


class HCNN3(CNN3):
	
	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded):
		
		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, only_thresholded)
		
		self.branch_size = branch_size
		self.reinforce = reinforce
		self.projection = projection
		
		self.layer1  = nn.Conv2d(3, 64, (3,3), padding = 'same', bias = False)
		self.layer2  = nn.BatchNorm2d(64)
		self.layer3  = nn.Conv2d(64, 64, (3,3), padding = 'same', bias = False)
		self.layer4  = nn.BatchNorm2d(64)
		self.layer5  = nn.MaxPool2d((2,2), stride = (2,2))

		self.layer6  = nn.Conv2d(64, 128, (3,3), padding = 'same', bias = False)
		self.layer7  = nn.BatchNorm2d(128)
		self.layer8  = nn.Conv2d(128, 128, (3,3), padding = 'same', bias = False)
		self.layer9  = nn.BatchNorm2d(128)
		self.layer10 = nn.MaxPool2d((2,2), stride = (2,2))
		
		self.layerb17 = nn.Linear(self.branch_size, self.dataset.num_c1)
		
		self.layerb27 = nn.Linear(self.branch_size, self.dataset.num_c2)
		
		self.layerb37 = nn.Linear(self.branch_size, self.dataset.num_c3)
		
		if self.projection:
			self.layerb27__ = nn.Linear(self.branch_size, self.dataset.num_c2)
			self.layerb37__ = nn.Linear(self.branch_size, self.dataset.num_c3)
		
		if self.reinforce and self.projection:
			self.layer_branch_2_final = self.layer_branch_2_final_reduce_prj
			self.layer_branch_3_final =	self.layer_branch_3_final_reduce_prj
		elif self.reinforce:
			self.layer_branch_2_final = self.layer_branch_2_final_reduce
			self.layer_branch_3_final =	self.layer_branch_3_final_reduce
		elif self.projection:
			self.layer_branch_2_final = self.layer_branch_2_final_prj
			self.layer_branch_3_final =	self.layer_branch_3_final_prj
		else:
			self.layer_branch_2_final = self.layer_branch_2_final_naive
			self.layer_branch_3_final = self.layer_branch_3_final_naive
	
	
	def forward_conv(self, x):
		return x
		
		
	def forward_branch(self, x):
		return x
		
		
	def layer_branch_2_final_naive(self, ort, prj, b1):
		return self.layerb27(ort) + self.layerb27__(prj)
		
		
	def layer_branch_2_final_reduce_prj(self, ort, prj, b1):
		return self.layerb27(ort) + self.layerb27__(prj) + self.c2_reinforce(b1.clone().detach())
		
	
	def layer_branch_2_final_prj(self, ort, prj, b1):
		return self.layerb27(ort) + self.c2_reinforce(b1.clone().detach())
		
		
	def layer_branch_3_final_naive(self, ort, prj, b2):
		return self.layerb37(ort) + self.layerb37__(prj)
		
		
	def layer_branch_3_final_reduce_prj(self, ort, prj, b2):
		return self.layerb37(ort) + self.layerb37__(prj) + self.c3_reinforce(b2.clone().detach())
		
		
	def layer_branch_3_final_prj(self, ort, prj, b2):
		return self.layerb37(ort) + self.c3_reinforce(b2.clone().detach())
	
		
	def forward(self, x):
		
		# block 1
		z = self.layer1(x)
		z = self.activation(z)
		z = self.layer2(z)
		z = self.layer3(z)
		z = self.activation(z)
		z = self.layer4(z)
		z = self.layer5(z)

		# block 2
		z = self.layer6(z)
		z = self.activation(z)
		z = self.layer7(z)
		z = self.layer8(z)
		z = self.activation(z)
		z = self.layer9(z)
		z = self.layer10(z)
		
		z = self.forward_conv(z)
		
		z = torch.flatten(z, start_dim = 1)
		
		z = self.forward_branch(z)

		# projections
		ort2, ort3, prj2, prj3 = self.project(z)

		# branch 1
		b1 = self.layerb17(z)
		
		# branch 2
		b2 = self.layer_branch_2_final(ort2, prj2, b1)

		# branch 3
		b3 = self.layer_branch_3_final(ort3, prj3, b2)

		return b1, b2, b3
		

class HCNN3_c0_b0(HCNN3):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)

		self.layerb_mid = nn.Linear(8*8*128, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)
			
	
	def forward_branch(self, z):
	
		z = self.layerb_mid(z)
		z = self.activation(z)
		
		return z
		
		
	def __str__(self):
		return "HCNN3_c0_b0"



class HCNN3_c0_b1(HCNN3):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)

		self.layerb11 = nn.Linear(8*8*128, self.branch_size, bias = False)
		self.layerb12 = nn.BatchNorm1d(self.branch_size)
		self.layerb13 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(self.branch_size, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_branch(self, z):

		z = self.layerb11(z)
		z = self.activation(z)
		z = self.layerb12(z)
		z = self.layerb13(z)
		z = self.layerb_mid(z)
		z = self.activation(z)

		return z
		

	def __str__(self):
		return "HCNN3_c0_b1"
		

		
class HCNN3_c0_b2(HCNN3):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)

		self.layerb11 = nn.Linear(8*8*128, self.branch_size, bias = False)
		self.layerb12 = nn.BatchNorm1d(self.branch_size)
		self.layerb13 = nn.Dropout(0.5)
		self.layerb14 = nn.Linear(self.branch_size, self.branch_size, bias = False)
		self.layerb15 = nn.BatchNorm1d(self.branch_size)
		self.layerb16 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(self.branch_size, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_branch(self, z):
	
		z = self.layerb11(z)
		z = self.activation(z)
		z = self.layerb12(z)
		z = self.layerb13(z)
		z = self.layerb14(z)
		z = self.activation(z)
		z = self.layerb15(z)
		z = self.layerb16(z)
		z = self.layerb_mid(z)
		z = self.activation(z)

		return z


	def __str__(self):
		return "HCNN3_c0_b2"
		
		
		
class HCNN3_c1_b0(HCNN3):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		self.layer10_1 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer10_2 = nn.BatchNorm2d(256)
		self.layer10_5 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb_mid = nn.Linear(4*4*256, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer10_1(z)
		z = self.activation(z)
		z = self.layer10_2(z)
		z = self.layer10_5(z)
		
		return z

	
	def forward_branch(self, z):
	
		z = self.layerb_mid(z)
		z = self.activation(z)
		
		return z


	def __str__(self):
		return "HCNN3_c1_b0"
		
		

class HCNN3_c1_b1(HCNN3):

	def __init__(self, learning_rate, momentum, nesterov, dataset, 
					epochs, every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		self.layer10_1 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer10_2 = nn.BatchNorm2d(256)
		self.layer10_5 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb11 = nn.Linear(4*4*256, self.branch_size, bias = False)
		self.layerb12 = nn.BatchNorm1d(self.branch_size)
		self.layerb13 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(self.branch_size, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer10_1(z)
		z = self.activation(z)
		z = self.layer10_2(z)
		z = self.layer10_5(z)
		
		return z

	
	def forward_branch(self, z):
		
		z = self.layerb11(z)
		z = self.activation(z)
		z = self.layerb12(z)
		z = self.layerb13(z)
		z = self.layerb_mid(z)
		z = self.activation(z)
		
		return z


	def __str__(self):
		return "HCNN3_c1_b1"
		
		
		
class HCNN3_c1_b2(HCNN3):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		self.layer10_1 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer10_2 = nn.BatchNorm2d(256)
		self.layer10_5 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb11 = nn.Linear(4*4*256, self.branch_size, bias = False)
		self.layerb12 = nn.BatchNorm1d(self.branch_size)
		self.layerb13 = nn.Dropout(0.5)
		self.layerb14 = nn.Linear(self.branch_size, self.branch_size, bias = False)
		self.layerb15 = nn.BatchNorm1d(self.branch_size)
		self.layerb16 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(self.branch_size, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer10_1(z)
		z = self.activation(z)
		z = self.layer10_2(z)
		z = self.layer10_5(z)
		
		return z

	def forward_branch(self, z):
	
		z = self.layerb11(z)
		z = self.activation(z)
		z = self.layerb12(z)
		z = self.layerb13(z)
		z = self.layerb14(z)
		z = self.activation(z)
		z = self.layerb15(z)
		z = self.layerb16(z)
		z = self.layerb_mid(z)
		z = self.activation(z)

		return z


	def __str__(self):
		return "HCNN3_c1_b2"	



class HCNN3_c2_b0(HCNN3):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		self.layer10_1 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer10_2 = nn.BatchNorm2d(256)
		self.layer10_3 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer10_4 = nn.BatchNorm2d(256)
		self.layer10_5 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb_mid = nn.Linear(4*4*256, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer10_1(z)
		z = self.activation(z)
		z = self.layer10_2(z)
		z = self.layer10_3(z)
		z = self.activation(z)
		z = self.layer10_4(z)
		z = self.layer10_5(z)
		
		return z

	
	def forward_branch(self, z):
	
		z = self.layerb_mid(z)
		z = self.activation(z)
		
		return z


	def __str__(self):
		return "HCNN3_c2_b0"	



class HCNN3_c2_b1(HCNN3):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		self.layer10_1 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer10_2 = nn.BatchNorm2d(256)
		self.layer10_3 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer10_4 = nn.BatchNorm2d(256)
		self.layer10_5 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb11 = nn.Linear(4*4*256, self.branch_size, bias = False)
		self.layerb12 = nn.BatchNorm1d(self.branch_size)
		self.layerb13 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(self.branch_size, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer10_1(z)
		z = self.activation(z)
		z = self.layer10_2(z)
		z = self.layer10_3(z)
		z = self.activation(z)
		z = self.layer10_4(z)
		z = self.layer10_5(z)
		
		return z

	
	def forward_branch(self, z):
		
		z = self.layerb11(z)
		z = self.activation(z)
		z = self.layerb12(z)
		z = self.layerb13(z)
		z = self.layerb_mid(z)
		z = self.activation(z)
		
		return z


	def __str__(self):
		return "HCNN3_c2_b1"	
	
		
		
class HCNN3_c2_b2(HCNN3):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		self.layer10_1 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer10_2 = nn.BatchNorm2d(256)
		self.layer10_3 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer10_4 = nn.BatchNorm2d(256)
		self.layer10_5 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb11 = nn.Linear(4*4*256, self.branch_size, bias = False)
		self.layerb12 = nn.BatchNorm1d(self.branch_size)
		self.layerb13 = nn.Dropout(0.5)
		self.layerb14 = nn.Linear(self.branch_size, self.branch_size, bias = False)
		self.layerb15 = nn.BatchNorm1d(self.branch_size)
		self.layerb16 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(self.branch_size, self.branch_size)
		
		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer10_1(z)
		z = self.activation(z)
		z = self.layer10_2(z)
		z = self.layer10_3(z)
		z = self.activation(z)
		z = self.layer10_4(z)
		z = self.layer10_5(z)
		
		return z

	
	def forward_branch(self, z):
		
		z = self.layerb11(z)
		z = self.activation(z)
		z = self.layerb12(z)
		z = self.layerb13(z)
		z = self.layerb14(z)
		z = self.activation(z)
		z = self.layerb15(z)
		z = self.layerb16(z)
		z = self.layerb_mid(z)
		z = self.activation(z)
		
		return z
		
		
	def __str__(self):
		return "HCNN3_c2_b2"
		

class HCNN3_c3_b0(HCNN3):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		self.layer10_1 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer10_2 = nn.BatchNorm2d(256)
		self.layer10_3 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer10_4 = nn.BatchNorm2d(256)
		self.layer10_5 = nn.MaxPool2d((2,2), stride = (2,2))
		
		self.layer10_6 = nn.Conv2d(256, 512, (3,3), padding = 'same', bias = False)
		self.layer10_7 = nn.BatchNorm2d(512)
		self.layer10_10 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb_mid = nn.Linear(2*2*512, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer10_1(z)
		z = self.activation(z)
		z = self.layer10_2(z)
		z = self.layer10_3(z)
		z = self.activation(z)
		z = self.layer10_4(z)
		z = self.layer10_5(z)
		
		z = self.layer10_6(z)
		z = self.activation(z)
		z = self.layer10_7(z)
		z = self.layer10_10(z)
		
		return z

	
	def forward_branch(self, z):
	
		z = self.layerb_mid(z)
		z = self.activation(z)
		
		return z


	def __str__(self):
		return "HCNN3_c3_b0"	



class HCNN3_c3_b1(HCNN3):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		self.layer10_1 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer10_2 = nn.BatchNorm2d(256)
		self.layer10_3 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer10_4 = nn.BatchNorm2d(256)
		self.layer10_5 = nn.MaxPool2d((2,2), stride = (2,2))
		
		self.layer10_6 = nn.Conv2d(256, 512, (3,3), padding = 'same', bias = False)
		self.layer10_7 = nn.BatchNorm2d(512)
		self.layer10_10 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb11 = nn.Linear(2*2*512, self.branch_size, bias = False)
		self.layerb12 = nn.BatchNorm1d(self.branch_size)
		self.layerb13 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(self.branch_size, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer10_1(z)
		z = self.activation(z)
		z = self.layer10_2(z)
		z = self.layer10_3(z)
		z = self.activation(z)
		z = self.layer10_4(z)
		z = self.layer10_5(z)
		
		z = self.layer10_6(z)
		z = self.activation(z)
		z = self.layer10_7(z)
		z = self.layer10_10(z)
		
		return z

	
	def forward_branch(self, z):
		
		z = self.layerb11(z)
		z = self.activation(z)
		z = self.layerb12(z)
		z = self.layerb13(z)
		z = self.layerb_mid(z)
		z = self.activation(z)
		
		return z


	def __str__(self):
		return "HCNN3_c3_b1"	
	
		
		
class HCNN3_c3_b2(HCNN3):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		self.layer10_1 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer10_2 = nn.BatchNorm2d(256)
		self.layer10_3 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer10_4 = nn.BatchNorm2d(256)
		self.layer10_5 = nn.MaxPool2d((2,2), stride = (2,2))
		
		self.layer10_6 = nn.Conv2d(256, 512, (3,3), padding = 'same', bias = False)
		self.layer10_7 = nn.BatchNorm2d(512)
		self.layer10_10 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb11 = nn.Linear(2*2*512, self.branch_size, bias = False)
		self.layerb12 = nn.BatchNorm1d(self.branch_size)
		self.layerb13 = nn.Dropout(0.5)
		self.layerb14 = nn.Linear(self.branch_size, self.branch_size, bias = False)
		self.layerb15 = nn.BatchNorm1d(self.branch_size)
		self.layerb16 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(self.branch_size, self.branch_size)
		
		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer10_1(z)
		z = self.activation(z)
		z = self.layer10_2(z)
		z = self.layer10_3(z)
		z = self.activation(z)
		z = self.layer10_4(z)
		z = self.layer10_5(z)
		
		z = self.layer10_6(z)
		z = self.activation(z)
		z = self.layer10_7(z)
		z = self.layer10_10(z)
		
		return z

	
	def forward_branch(self, z):
		
		z = self.layerb11(z)
		z = self.activation(z)
		z = self.layerb12(z)
		z = self.layerb13(z)
		z = self.layerb14(z)
		z = self.activation(z)
		z = self.layerb15(z)
		z = self.layerb16(z)
		z = self.layerb_mid(z)
		z = self.activation(z)
		
		return z
		
		
	def __str__(self):
		return "HCNN3_c3_b2"
		
		
class HCNN3_c4_b0(HCNN3):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		self.layer10_1 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer10_2 = nn.BatchNorm2d(256)
		self.layer10_3 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer10_4 = nn.BatchNorm2d(256)
		self.layer10_5 = nn.MaxPool2d((2,2), stride = (2,2))
		
		self.layer10_6 = nn.Conv2d(256, 512, (3,3), padding = 'same', bias = False)
		self.layer10_7 = nn.BatchNorm2d(512)
		self.layer10_8 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer10_9 = nn.BatchNorm2d(512)
		self.layer10_10 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb_mid = nn.Linear(2*2*512, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer10_1(z)
		z = self.activation(z)
		z = self.layer10_2(z)
		z = self.layer10_3(z)
		z = self.activation(z)
		z = self.layer10_4(z)
		z = self.layer10_5(z)
		
		z = self.layer10_6(z)
		z = self.activation(z)
		z = self.layer10_7(z)
		z = self.layer10_8(z)
		z = self.activation(z)
		z = self.layer10_9(z)
		z = self.layer10_10(z)
		
		return z

	
	def forward_branch(self, z):
	
		z = self.layerb_mid(z)
		z = self.activation(z)
		
		return z


	def __str__(self):
		return "HCNN3_c4_b0"	



class HCNN3_c4_b1(HCNN3):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		self.layer10_1 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer10_2 = nn.BatchNorm2d(256)
		self.layer10_3 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer10_4 = nn.BatchNorm2d(256)
		self.layer10_5 = nn.MaxPool2d((2,2), stride = (2,2))
		
		self.layer10_6 = nn.Conv2d(256, 512, (3,3), padding = 'same', bias = False)
		self.layer10_7 = nn.BatchNorm2d(512)
		self.layer10_8 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer10_9 = nn.BatchNorm2d(512)
		self.layer10_10 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb11 = nn.Linear(2*2*512, self.branch_size, bias = False)
		self.layerb12 = nn.BatchNorm1d(self.branch_size)
		self.layerb13 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(self.branch_size, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer10_1(z)
		z = self.activation(z)
		z = self.layer10_2(z)
		z = self.layer10_3(z)
		z = self.activation(z)
		z = self.layer10_4(z)
		z = self.layer10_5(z)
		
		z = self.layer10_6(z)
		z = self.activation(z)
		z = self.layer10_7(z)
		z = self.layer10_8(z)
		z = self.activation(z)
		z = self.layer10_9(z)
		z = self.layer10_10(z)
		
		return z

	
	def forward_branch(self, z):
		
		z = self.layerb11(z)
		z = self.activation(z)
		z = self.layerb12(z)
		z = self.layerb13(z)
		z = self.layerb_mid(z)
		z = self.activation(z)
		
		return z


	def __str__(self):
		return "HCNN3_c4_b1"	
	
		
		
class HCNN3_c4_b2(HCNN3):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		self.layer10_1 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer10_2 = nn.BatchNorm2d(256)
		self.layer10_3 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer10_4 = nn.BatchNorm2d(256)
		self.layer10_5 = nn.MaxPool2d((2,2), stride = (2,2))
		
		self.layer10_6 = nn.Conv2d(256, 512, (3,3), padding = 'same', bias = False)
		self.layer10_7 = nn.BatchNorm2d(512)
		self.layer10_8 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer10_9 = nn.BatchNorm2d(512)
		self.layer10_10 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb11 = nn.Linear(2*2*512, self.branch_size, bias = False)
		self.layerb12 = nn.BatchNorm1d(self.branch_size)
		self.layerb13 = nn.Dropout(0.5)
		self.layerb14 = nn.Linear(self.branch_size, self.branch_size, bias = False)
		self.layerb15 = nn.BatchNorm1d(self.branch_size)
		self.layerb16 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(self.branch_size, self.branch_size)
		
		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer10_1(z)
		z = self.activation(z)
		z = self.layer10_2(z)
		z = self.layer10_3(z)
		z = self.activation(z)
		z = self.layer10_4(z)
		z = self.layer10_5(z)
		
		z = self.layer10_6(z)
		z = self.activation(z)
		z = self.layer10_7(z)
		z = self.layer10_8(z)
		z = self.activation(z)
		z = self.layer10_9(z)
		z = self.layer10_10(z)
		
		return z

	
	def forward_branch(self, z):
		
		z = self.layerb11(z)
		z = self.activation(z)
		z = self.layerb12(z)
		z = self.layerb13(z)
		z = self.layerb14(z)
		z = self.activation(z)
		z = self.layerb15(z)
		z = self.layerb16(z)
		z = self.layerb_mid(z)
		z = self.activation(z)
		
		return z
		
		
	def __str__(self):
		return "HCNN3_c4_b2"
