import cnn3

from cnn3 import torch, optim
from cnn3 import CNN3
from cnn3 import nn
from cnn3 import np


class HVGG(CNN3):
	
	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded):
		
		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, only_thresholded)
		
		self.branch_size = branch_size
		self.reinforce = reinforce
		self.projection = projection
		
		# Block 1
		self.layer1  = nn.Conv2d(1, 64, (3,3), padding = 'same', bias = False)
		self.layer2  = nn.BatchNorm2d(64)
		self.layer3  = nn.Conv2d(64, 64, (3,3), padding = 'same', bias = False)
		self.layer4  = nn.BatchNorm2d(64)
		self.layer5  = nn.MaxPool2d((2,2), stride = (2,2))


		# Block 2
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
			self.layer_branch_2_final = self.layer_branch_2_final_reinforce_prj
			self.layer_branch_3_final =	self.layer_branch_3_final_reinforce_prj
		elif self.reinforce:
			self.layer_branch_2_final = self.layer_branch_2_final_reinforce
			self.layer_branch_3_final =	self.layer_branch_3_final_reinforce
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
		return self.layerb27(ort + prj)
		
		
	def layer_branch_2_final_reinforce_prj(self, ort, prj, b1):
		return self.layerb27(ort) + self.layerb27__(prj) + self.c2_reinforce(b1.clone().detach())
		
		
	def layer_branch_2_final_prj(self, ort, prj, b1):
		return self.layerb27(ort) + self.layerb27__(prj)
		
	
	def layer_branch_2_final_reinforce(self, ort, prj, b1):
		return self.layerb27(ort + prj) + self.c2_reinforce(b1.clone().detach())
		
		
	def layer_branch_3_final_naive(self, ort, prj, b2):
		return self.layerb37(ort + prj)
		
		
	def layer_branch_3_final_reinforce_prj(self, ort, prj, b2):
		return self.layerb37(ort) + self.layerb37__(prj) + self.c3_reinforce(b2.clone().detach())
		
		
	def layer_branch_3_final_prj(self, ort, prj, b2):
		return self.layerb37(ort) + self.layerb37__(prj)
		
		
	def layer_branch_3_final_reinforce(self, ort, prj, b2):
		return self.layerb37(ort + prj) + self.c3_reinforce(b2.clone().detach())
		
	
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
		
	
class HVGG_c0_b1(HVGG):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):
					
		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)

		self.layerb11 = nn.Linear(7*7*128, self.branch_size, bias = False)
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
		return "HVGG_c0_b1"
		
		
class HVGG_c0_b2(HVGG):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)

		self.layerb11 = nn.Linear(7*7*128, self.branch_size, bias = False)
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
		return "HVGG_c0_b2"
		

class HVGG_c1_b0_16(HVGG):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		# Block 3
		self.layer11 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer12 = nn.BatchNorm2d(256)
		self.layer13 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer14 = nn.BatchNorm2d(256)
		self.layer15 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer16 = nn.BatchNorm2d(256)
		self.layer17 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb_mid = nn.Linear(3*3*256, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer11(z)
		z = self.activation(z)
		z = self.layer12(z)
		z = self.layer13(z)
		z = self.activation(z)
		z = self.layer14(z)
		z = self.layer15(z)
		z = self.activation(z)
		z = self.layer16(z)
		z = self.layer17(z)
		
		return z

	
	def forward_branch(self, z):
	
		z = self.layerb_mid(z)
		z = self.activation(z)
		
		return z


	def __str__(self):
		return "HVGG_c1_b0_16"
		
		
class HVGG_c1_b0_19(HVGG):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		# Block 3
		self.layer11 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer12 = nn.BatchNorm2d(256)
		self.layer13 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer14 = nn.BatchNorm2d(256)
		self.layer15 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer16 = nn.BatchNorm2d(256)
		self.layer17 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer18 = nn.BatchNorm2d(256)
		self.layer19 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb_mid = nn.Linear(3*3*256, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer11(z)
		z = self.activation(z)
		z = self.layer12(z)
		z = self.layer13(z)
		z = self.activation(z)
		z = self.layer14(z)
		z = self.layer15(z)
		z = self.activation(z)
		z = self.layer16(z)
		z = self.layer17(z)
		z = self.activation(z)
		z = self.layer18(z)
		z = self.layer19(z)
		
		return z

	
	def forward_branch(self, z):
	
		z = self.layerb_mid(z)
		z = self.activation(z)
		
		return z


	def __str__(self):
		return "HVGG_c1_b0_19"
		
		
class HVGG_c1_b1_16(HVGG):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		# Block 3
		self.layer11 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer12 = nn.BatchNorm2d(256)
		self.layer13 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer14 = nn.BatchNorm2d(256)
		self.layer15 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer16 = nn.BatchNorm2d(256)
		self.layer17 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb11 = nn.Linear(3*3*256, self.branch_size, bias = False)
		self.layerb12 = nn.BatchNorm1d(self.branch_size)
		self.layerb13 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(self.branch_size, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer11(z)
		z = self.activation(z)
		z = self.layer12(z)
		z = self.layer13(z)
		z = self.activation(z)
		z = self.layer14(z)
		z = self.layer15(z)
		z = self.activation(z)
		z = self.layer16(z)
		z = self.layer17(z)
		
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
		return "HVGG_c1_b1_16"
		
		
class HVGG_c1_b1_19(HVGG):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		# Block 3
		self.layer11 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer12 = nn.BatchNorm2d(256)
		self.layer13 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer14 = nn.BatchNorm2d(256)
		self.layer15 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer16 = nn.BatchNorm2d(256)
		self.layer17 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer18 = nn.BatchNorm2d(256)
		self.layer19 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb11 = nn.Linear(3*3*256, self.branch_size, bias = False)
		self.layerb12 = nn.BatchNorm1d(self.branch_size)
		self.layerb13 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(self.branch_size, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer11(z)
		z = self.activation(z)
		z = self.layer12(z)
		z = self.layer13(z)
		z = self.activation(z)
		z = self.layer14(z)
		z = self.layer15(z)
		z = self.activation(z)
		z = self.layer16(z)
		z = self.layer17(z)
		z = self.activation(z)
		z = self.layer18(z)
		z = self.layer19(z)
		
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
		return "HVGG_c1_b1_19"
		
		
class HVGG_c1_b2_16(HVGG):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		# Block 3
		self.layer11 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer12 = nn.BatchNorm2d(256)
		self.layer13 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer14 = nn.BatchNorm2d(256)
		self.layer15 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer16 = nn.BatchNorm2d(256)
		self.layer17 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb11 = nn.Linear(3*3*256, self.branch_size, bias = False)
		self.layerb12 = nn.BatchNorm1d(self.branch_size)
		self.layerb13 = nn.Dropout(0.5)
		self.layerb14 = nn.Linear(self.branch_size, self.branch_size, bias = False)
		self.layerb15 = nn.BatchNorm1d(self.branch_size)
		self.layerb16 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(self.branch_size, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer11(z)
		z = self.activation(z)
		z = self.layer12(z)
		z = self.layer13(z)
		z = self.activation(z)
		z = self.layer14(z)
		z = self.layer15(z)
		z = self.activation(z)
		z = self.layer16(z)
		z = self.layer17(z)
		
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
		return "HVGG_c1_b2_16"
		
		
class HVGG_c1_b2_19(HVGG):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		# Block 3
		self.layer11 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer12 = nn.BatchNorm2d(256)
		self.layer13 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer14 = nn.BatchNorm2d(256)
		self.layer15 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer16 = nn.BatchNorm2d(256)
		self.layer17 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer18 = nn.BatchNorm2d(256)
		self.layer19 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb11 = nn.Linear(3*3*256, self.branch_size, bias = False)
		self.layerb12 = nn.BatchNorm1d(self.branch_size)
		self.layerb13 = nn.Dropout(0.5)
		self.layerb14 = nn.Linear(self.branch_size, self.branch_size, bias = False)
		self.layerb15 = nn.BatchNorm1d(self.branch_size)
		self.layerb16 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(self.branch_size, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer11(z)
		z = self.activation(z)
		z = self.layer12(z)
		z = self.layer13(z)
		z = self.activation(z)
		z = self.layer14(z)
		z = self.layer15(z)
		z = self.activation(z)
		z = self.layer16(z)
		z = self.layer17(z)
		z = self.activation(z)
		z = self.layer18(z)
		z = self.layer19(z)
		
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
		return "HVGG_c1_b2_19"


class HVGG_c2_b0_16(HVGG):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		# Block 3
		self.layer11 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer12 = nn.BatchNorm2d(256)
		self.layer13 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer14 = nn.BatchNorm2d(256)
		self.layer15 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer16 = nn.BatchNorm2d(256)
		self.layer17 = nn.MaxPool2d((2,2), stride = (2,2))
		
		# Block 4
		self.layer18 = nn.Conv2d(256, 512, (3,3), padding = 'same', bias = False)
		self.layer19 = nn.BatchNorm2d(512)
		self.layer20 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer21 = nn.BatchNorm2d(512)
		self.layer22 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer23 = nn.BatchNorm2d(512)
		self.layer24 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb_mid = nn.Linear(1*1*512, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer11(z)
		z = self.activation(z)
		z = self.layer12(z)
		z = self.layer13(z)
		z = self.activation(z)
		z = self.layer14(z)
		z = self.layer15(z)
		z = self.activation(z)
		z = self.layer16(z)
		z = self.layer17(z)
		
		z = self.layer18(z)
		z = self.activation(z)
		z = self.layer19(z)
		z = self.layer20(z)
		z = self.activation(z)
		z = self.layer21(z)
		z = self.layer22(z)
		z = self.activation(z)
		z = self.layer23(z)
		z = self.layer24(z)
		
		return z

	
	def forward_branch(self, z):
	
		z = self.layerb_mid(z)
		z = self.activation(z)
		
		return z


	def __str__(self):
		return "HVGG_c2_b0_16"
		
		
class HVGG_c2_b0_19(HVGG):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		# Block 3
		self.layer11 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer12 = nn.BatchNorm2d(256)
		self.layer13 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer14 = nn.BatchNorm2d(256)
		self.layer15 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer16 = nn.BatchNorm2d(256)
		self.layer17 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer18 = nn.BatchNorm2d(256)
		self.layer19 = nn.MaxPool2d((2,2), stride = (2,2))
		
		# Block 4
		self.layer20 = nn.Conv2d(256, 512, (3,3), padding = 'same', bias = False)
		self.layer21 = nn.BatchNorm2d(512)
		self.layer22 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer23 = nn.BatchNorm2d(512)
		self.layer24 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer25 = nn.BatchNorm2d(512)
		self.layer26 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer27 = nn.BatchNorm2d(512)
		self.layer28 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb_mid = nn.Linear(1*1*512, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer11(z)
		z = self.activation(z)
		z = self.layer12(z)
		z = self.layer13(z)
		z = self.activation(z)
		z = self.layer14(z)
		z = self.layer15(z)
		z = self.activation(z)
		z = self.layer16(z)
		z = self.layer17(z)
		z = self.activation(z)
		z = self.layer18(z)
		z = self.layer19(z)
		
		z = self.layer20(z)
		z = self.activation(z)
		z = self.layer21(z)
		z = self.layer22(z)
		z = self.activation(z)
		z = self.layer23(z)
		z = self.layer24(z)
		z = self.activation(z)
		z = self.layer25(z)
		z = self.layer26(z)
		z = self.activation(z)
		z = self.layer27(z)
		z = self.layer28(z)
		
		return z

	
	def forward_branch(self, z):
	
		z = self.layerb_mid(z)
		z = self.activation(z)
		
		return z


	def __str__(self):
		return "HVGG_c2_b0_19"
		
		
class HVGG_c2_b1_16(HVGG):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		# Block 3
		self.layer11 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer12 = nn.BatchNorm2d(256)
		self.layer13 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer14 = nn.BatchNorm2d(256)
		self.layer15 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer16 = nn.BatchNorm2d(256)
		self.layer17 = nn.MaxPool2d((2,2), stride = (2,2))
		
		# Block 4
		self.layer18 = nn.Conv2d(256, 512, (3,3), padding = 'same', bias = False)
		self.layer19 = nn.BatchNorm2d(512)
		self.layer20 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer21 = nn.BatchNorm2d(512)
		self.layer22 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer23 = nn.BatchNorm2d(512)
		self.layer24 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb11 = nn.Linear(1*1*512, self.branch_size, bias = False)
		self.layerb12 = nn.BatchNorm1d(self.branch_size)
		self.layerb13 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(self.branch_size, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer11(z)
		z = self.activation(z)
		z = self.layer12(z)
		z = self.layer13(z)
		z = self.activation(z)
		z = self.layer14(z)
		z = self.layer15(z)
		z = self.activation(z)
		z = self.layer16(z)
		z = self.layer17(z)
		
		z = self.layer18(z)
		z = self.activation(z)
		z = self.layer19(z)
		z = self.layer20(z)
		z = self.activation(z)
		z = self.layer21(z)
		z = self.layer22(z)
		z = self.activation(z)
		z = self.layer23(z)
		z = self.layer24(z)
		
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
		return "HVGG_c2_b1_16"
		
		
class HVGG_c2_b1_19(HVGG):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		# Block 3
		self.layer11 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer12 = nn.BatchNorm2d(256)
		self.layer13 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer14 = nn.BatchNorm2d(256)
		self.layer15 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer16 = nn.BatchNorm2d(256)
		self.layer17 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer18 = nn.BatchNorm2d(256)
		self.layer19 = nn.MaxPool2d((2,2), stride = (2,2))
		
		# Block 4
		self.layer20 = nn.Conv2d(256, 512, (3,3), padding = 'same', bias = False)
		self.layer21 = nn.BatchNorm2d(512)
		self.layer22 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer23 = nn.BatchNorm2d(512)
		self.layer24 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer25 = nn.BatchNorm2d(512)
		self.layer26 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer27 = nn.BatchNorm2d(512)
		self.layer28 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb11 = nn.Linear(1*1*512, self.branch_size, bias = False)
		self.layerb12 = nn.BatchNorm1d(self.branch_size)
		self.layerb13 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(self.branch_size, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer11(z)
		z = self.activation(z)
		z = self.layer12(z)
		z = self.layer13(z)
		z = self.activation(z)
		z = self.layer14(z)
		z = self.layer15(z)
		z = self.activation(z)
		z = self.layer16(z)
		z = self.layer17(z)
		z = self.activation(z)
		z = self.layer18(z)
		z = self.layer19(z)
		
		z = self.layer20(z)
		z = self.activation(z)
		z = self.layer21(z)
		z = self.layer22(z)
		z = self.activation(z)
		z = self.layer23(z)
		z = self.layer24(z)
		z = self.activation(z)
		z = self.layer25(z)
		z = self.layer26(z)
		z = self.activation(z)
		z = self.layer27(z)
		z = self.layer28(z)
		
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
		return "HVGG_c2_b1_19"
		
		
class HVGG_c2_b2_16(HVGG):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		# Block 3
		self.layer11 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer12 = nn.BatchNorm2d(256)
		self.layer13 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer14 = nn.BatchNorm2d(256)
		self.layer15 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer16 = nn.BatchNorm2d(256)
		self.layer17 = nn.MaxPool2d((2,2), stride = (2,2))
		
		# Block 4
		self.layer18 = nn.Conv2d(256, 512, (3,3), padding = 'same', bias = False)
		self.layer19 = nn.BatchNorm2d(512)
		self.layer20 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer21 = nn.BatchNorm2d(512)
		self.layer22 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer23 = nn.BatchNorm2d(512)
		self.layer24 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb11 = nn.Linear(1*1*512, self.branch_size, bias = False)
		self.layerb12 = nn.BatchNorm1d(self.branch_size)
		self.layerb13 = nn.Dropout(0.5)
		self.layerb14 = nn.Linear(self.branch_size, self.branch_size, bias = False)
		self.layerb15 = nn.BatchNorm1d(self.branch_size)
		self.layerb16 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(self.branch_size, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer11(z)
		z = self.activation(z)
		z = self.layer12(z)
		z = self.layer13(z)
		z = self.activation(z)
		z = self.layer14(z)
		z = self.layer15(z)
		z = self.activation(z)
		z = self.layer16(z)
		z = self.layer17(z)
		
		z = self.layer18(z)
		z = self.activation(z)
		z = self.layer19(z)
		z = self.layer20(z)
		z = self.activation(z)
		z = self.layer21(z)
		z = self.layer22(z)
		z = self.activation(z)
		z = self.layer23(z)
		z = self.layer24(z)
		
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
		return "HVGG_c2_b2_16"
		
		
class HVGG_c2_b2_19(HVGG):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		# Block 3
		self.layer11 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer12 = nn.BatchNorm2d(256)
		self.layer13 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer14 = nn.BatchNorm2d(256)
		self.layer15 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer16 = nn.BatchNorm2d(256)
		self.layer17 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer18 = nn.BatchNorm2d(256)
		self.layer19 = nn.MaxPool2d((2,2), stride = (2,2))
		
		# Block 4
		self.layer20 = nn.Conv2d(256, 512, (3,3), padding = 'same', bias = False)
		self.layer21 = nn.BatchNorm2d(512)
		self.layer22 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer23 = nn.BatchNorm2d(512)
		self.layer24 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer25 = nn.BatchNorm2d(512)
		self.layer26 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer27 = nn.BatchNorm2d(512)
		self.layer28 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb11 = nn.Linear(1*1*512, self.branch_size, bias = False)
		self.layerb12 = nn.BatchNorm1d(self.branch_size)
		self.layerb13 = nn.Dropout(0.5)
		self.layerb14 = nn.Linear(self.branch_size, self.branch_size, bias = False)
		self.layerb15 = nn.BatchNorm1d(self.branch_size)
		self.layerb16 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(self.branch_size, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer11(z)
		z = self.activation(z)
		z = self.layer12(z)
		z = self.layer13(z)
		z = self.activation(z)
		z = self.layer14(z)
		z = self.layer15(z)
		z = self.activation(z)
		z = self.layer16(z)
		z = self.layer17(z)
		z = self.activation(z)
		z = self.layer18(z)
		z = self.layer19(z)
		
		z = self.layer20(z)
		z = self.activation(z)
		z = self.layer21(z)
		z = self.layer22(z)
		z = self.activation(z)
		z = self.layer23(z)
		z = self.layer24(z)
		z = self.activation(z)
		z = self.layer25(z)
		z = self.layer26(z)
		z = self.activation(z)
		z = self.layer27(z)
		z = self.layer28(z)
		
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
		return "HVGG_c2_b2_19"
		

class HVGG_c3_b0_16(HVGG):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		# Block 3
		self.layer11 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer12 = nn.BatchNorm2d(256)
		self.layer13 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer14 = nn.BatchNorm2d(256)
		self.layer15 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer16 = nn.BatchNorm2d(256)
		self.layer17 = nn.MaxPool2d((2,2), stride = (2,2))
		
		# Block 4
		self.layer18 = nn.Conv2d(256, 512, (3,3), padding = 'same', bias = False)
		self.layer19 = nn.BatchNorm2d(512)
		self.layer20 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer21 = nn.BatchNorm2d(512)
		self.layer22 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer23 = nn.BatchNorm2d(512)
		self.layer24 = nn.MaxPool2d((2,2), stride = (2,2))
		
		# Block 5
		self.layer25 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer26 = nn.BatchNorm2d(512)
		self.layer27 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer28 = nn.BatchNorm2d(512)
		self.layer29 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer30 = nn.BatchNorm2d(512)

		self.layerb_mid = nn.Linear(1*1*512, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer11(z)
		z = self.activation(z)
		z = self.layer12(z)
		z = self.layer13(z)
		z = self.activation(z)
		z = self.layer14(z)
		z = self.layer15(z)
		z = self.activation(z)
		z = self.layer16(z)
		z = self.layer17(z)
		
		z = self.layer18(z)
		z = self.activation(z)
		z = self.layer19(z)
		z = self.layer20(z)
		z = self.activation(z)
		z = self.layer21(z)
		z = self.layer22(z)
		z = self.activation(z)
		z = self.layer23(z)
		z = self.layer24(z)
		
		z = self.layer25(z)
		z = self.activation(z)
		z = self.layer26(z)
		z = self.layer27(z)
		z = self.activation(z)
		z = self.layer28(z)
		z = self.layer29(z)
		z = self.activation(z)
		z = self.layer30(z)
		
		return z

	
	def forward_branch(self, z):
	
		z = self.layerb_mid(z)
		z = self.activation(z)
		
		return z


	def __str__(self):
		return "HVGG_c3_b0_16"
		
		
class HVGG_c3_b0_19(HVGG):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		# Block 3
		self.layer11 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer12 = nn.BatchNorm2d(256)
		self.layer13 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer14 = nn.BatchNorm2d(256)
		self.layer15 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer16 = nn.BatchNorm2d(256)
		self.layer17 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer18 = nn.BatchNorm2d(256)
		self.layer19 = nn.MaxPool2d((2,2), stride = (2,2))
		
		# Block 4
		self.layer20 = nn.Conv2d(256, 512, (3,3), padding = 'same', bias = False)
		self.layer21 = nn.BatchNorm2d(512)
		self.layer22 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer23 = nn.BatchNorm2d(512)
		self.layer24 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer25 = nn.BatchNorm2d(512)
		self.layer26 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer27 = nn.BatchNorm2d(512)
		self.layer28 = nn.MaxPool2d((2,2), stride = (2,2))
		
		# Block 5
		self.layer29 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer30 = nn.BatchNorm2d(512)
		self.layer31 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer32 = nn.BatchNorm2d(512)
		self.layer33 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer34 = nn.BatchNorm2d(512)
		self.layer35 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer36 = nn.BatchNorm2d(512)

		self.layerb_mid = nn.Linear(1*1*512, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer11(z)
		z = self.activation(z)
		z = self.layer12(z)
		z = self.layer13(z)
		z = self.activation(z)
		z = self.layer14(z)
		z = self.layer15(z)
		z = self.activation(z)
		z = self.layer16(z)
		z = self.layer17(z)
		z = self.activation(z)
		z = self.layer18(z)
		z = self.layer19(z)
		
		z = self.layer20(z)
		z = self.activation(z)
		z = self.layer21(z)
		z = self.layer22(z)
		z = self.activation(z)
		z = self.layer23(z)
		z = self.layer24(z)
		z = self.activation(z)
		z = self.layer25(z)
		z = self.layer26(z)
		z = self.activation(z)
		z = self.layer27(z)
		z = self.layer28(z)
		
		z = self.layer29(z)
		z = self.activation(z)
		z = self.layer30(z)
		z = self.layer31(z)
		z = self.activation(z)
		z = self.layer32(z)
		z = self.layer33(z)
		z = self.activation(z)
		z = self.layer34(z)
		z = self.layer35(z)
		z = self.activation(z)
		z = self.layer36(z)
		
		return z

	
	def forward_branch(self, z):
	
		z = self.layerb_mid(z)
		z = self.activation(z)
		
		return z


	def __str__(self):
		return "HVGG_c3_b0_19"
		
		
class HVGG_c3_b1_16(HVGG):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		# Block 3
		self.layer11 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer12 = nn.BatchNorm2d(256)
		self.layer13 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer14 = nn.BatchNorm2d(256)
		self.layer15 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer16 = nn.BatchNorm2d(256)
		self.layer17 = nn.MaxPool2d((2,2), stride = (2,2))
		
		# Block 4
		self.layer18 = nn.Conv2d(256, 512, (3,3), padding = 'same', bias = False)
		self.layer19 = nn.BatchNorm2d(512)
		self.layer20 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer21 = nn.BatchNorm2d(512)
		self.layer22 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer23 = nn.BatchNorm2d(512)
		self.layer24 = nn.MaxPool2d((2,2), stride = (2,2))
		
		# Block 5
		self.layer25 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer26 = nn.BatchNorm2d(512)
		self.layer27 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer28 = nn.BatchNorm2d(512)
		self.layer29 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer30 = nn.BatchNorm2d(512)

		self.layerb11 = nn.Linear(1*1*512, self.branch_size, bias = False)
		self.layerb12 = nn.BatchNorm1d(self.branch_size)
		self.layerb13 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(self.branch_size, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer11(z)
		z = self.activation(z)
		z = self.layer12(z)
		z = self.layer13(z)
		z = self.activation(z)
		z = self.layer14(z)
		z = self.layer15(z)
		z = self.activation(z)
		z = self.layer16(z)
		z = self.layer17(z)
		
		z = self.layer18(z)
		z = self.activation(z)
		z = self.layer19(z)
		z = self.layer20(z)
		z = self.activation(z)
		z = self.layer21(z)
		z = self.layer22(z)
		z = self.activation(z)
		z = self.layer23(z)
		z = self.layer24(z)
		
		z = self.layer25(z)
		z = self.activation(z)
		z = self.layer26(z)
		z = self.layer27(z)
		z = self.activation(z)
		z = self.layer28(z)
		z = self.layer29(z)
		z = self.activation(z)
		z = self.layer30(z)
		
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
		return "HVGG_c2_b1_16"
		
		
class HVGG_c3_b1_19(HVGG):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		# Block 3
		self.layer11 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer12 = nn.BatchNorm2d(256)
		self.layer13 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer14 = nn.BatchNorm2d(256)
		self.layer15 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer16 = nn.BatchNorm2d(256)
		self.layer17 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer18 = nn.BatchNorm2d(256)
		self.layer19 = nn.MaxPool2d((2,2), stride = (2,2))
		
		# Block 4
		self.layer20 = nn.Conv2d(256, 512, (3,3), padding = 'same', bias = False)
		self.layer21 = nn.BatchNorm2d(512)
		self.layer22 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer23 = nn.BatchNorm2d(512)
		self.layer24 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer25 = nn.BatchNorm2d(512)
		self.layer26 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer27 = nn.BatchNorm2d(512)
		self.layer28 = nn.MaxPool2d((2,2), stride = (2,2))
		
		# Block 5
		self.layer29 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer30 = nn.BatchNorm2d(512)
		self.layer31 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer32 = nn.BatchNorm2d(512)
		self.layer33 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer34 = nn.BatchNorm2d(512)
		self.layer35 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer36 = nn.BatchNorm2d(512)

		self.layerb11 = nn.Linear(1*1*512, self.branch_size, bias = False)
		self.layerb12 = nn.BatchNorm1d(self.branch_size)
		self.layerb13 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(self.branch_size, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer11(z)
		z = self.activation(z)
		z = self.layer12(z)
		z = self.layer13(z)
		z = self.activation(z)
		z = self.layer14(z)
		z = self.layer15(z)
		z = self.activation(z)
		z = self.layer16(z)
		z = self.layer17(z)
		z = self.activation(z)
		z = self.layer18(z)
		z = self.layer19(z)
		
		z = self.layer20(z)
		z = self.activation(z)
		z = self.layer21(z)
		z = self.layer22(z)
		z = self.activation(z)
		z = self.layer23(z)
		z = self.layer24(z)
		z = self.activation(z)
		z = self.layer25(z)
		z = self.layer26(z)
		z = self.activation(z)
		z = self.layer27(z)
		z = self.layer28(z)
		
		z = self.layer29(z)
		z = self.activation(z)
		z = self.layer30(z)
		z = self.layer31(z)
		z = self.activation(z)
		z = self.layer32(z)
		z = self.layer33(z)
		z = self.activation(z)
		z = self.layer34(z)
		z = self.layer35(z)
		z = self.activation(z)
		z = self.layer36(z)
		
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
		return "HVGG_c3_b1_19"
		
		
class HVGG_c3_b2_16(HVGG):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		# Block 3
		self.layer11 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer12 = nn.BatchNorm2d(256)
		self.layer13 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer14 = nn.BatchNorm2d(256)
		self.layer15 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer16 = nn.BatchNorm2d(256)
		self.layer17 = nn.MaxPool2d((2,2), stride = (2,2))
		
		# Block 4
		self.layer18 = nn.Conv2d(256, 512, (3,3), padding = 'same', bias = False)
		self.layer19 = nn.BatchNorm2d(512)
		self.layer20 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer21 = nn.BatchNorm2d(512)
		self.layer22 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer23 = nn.BatchNorm2d(512)
		self.layer24 = nn.MaxPool2d((2,2), stride = (2,2))
		
		# Block 5
		self.layer25 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer26 = nn.BatchNorm2d(512)
		self.layer27 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer28 = nn.BatchNorm2d(512)
		self.layer29 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer30 = nn.BatchNorm2d(512)

		self.layerb11 = nn.Linear(1*1*512, self.branch_size, bias = False)
		self.layerb12 = nn.BatchNorm1d(self.branch_size)
		self.layerb13 = nn.Dropout(0.5)
		self.layerb14 = nn.Linear(self.branch_size, self.branch_size, bias = False)
		self.layerb15 = nn.BatchNorm1d(self.branch_size)
		self.layerb16 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(self.branch_size, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer11(z)
		z = self.activation(z)
		z = self.layer12(z)
		z = self.layer13(z)
		z = self.activation(z)
		z = self.layer14(z)
		z = self.layer15(z)
		z = self.activation(z)
		z = self.layer16(z)
		z = self.layer17(z)
		
		z = self.layer18(z)
		z = self.activation(z)
		z = self.layer19(z)
		z = self.layer20(z)
		z = self.activation(z)
		z = self.layer21(z)
		z = self.layer22(z)
		z = self.activation(z)
		z = self.layer23(z)
		z = self.layer24(z)
		
		z = self.layer25(z)
		z = self.activation(z)
		z = self.layer26(z)
		z = self.layer27(z)
		z = self.activation(z)
		z = self.layer28(z)
		z = self.layer29(z)
		z = self.activation(z)
		z = self.layer30(z)
		
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
		return "HVGG_c3_b2_16"
		
		
class HVGG_c3_b2_19(HVGG):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, 
					every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean', branch_size = 512, reinforce = False, projection = False, only_thresholded = False):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction, branch_size, reinforce, projection, only_thresholded)
		
		# Block 3
		self.layer11 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer12 = nn.BatchNorm2d(256)
		self.layer13 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer14 = nn.BatchNorm2d(256)
		self.layer15 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer16 = nn.BatchNorm2d(256)
		self.layer17 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer18 = nn.BatchNorm2d(256)
		self.layer19 = nn.MaxPool2d((2,2), stride = (2,2))
		
		# Block 4
		self.layer20 = nn.Conv2d(256, 512, (3,3), padding = 'same', bias = False)
		self.layer21 = nn.BatchNorm2d(512)
		self.layer22 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer23 = nn.BatchNorm2d(512)
		self.layer24 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer25 = nn.BatchNorm2d(512)
		self.layer26 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer27 = nn.BatchNorm2d(512)
		self.layer28 = nn.MaxPool2d((2,2), stride = (2,2))
		
		# Block 5
		self.layer29 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer30 = nn.BatchNorm2d(512)
		self.layer31 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer32 = nn.BatchNorm2d(512)
		self.layer33 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer34 = nn.BatchNorm2d(512)
		self.layer35 = nn.Conv2d(512, 512, (3,3), padding = 'same', bias = False)
		self.layer36 = nn.BatchNorm2d(512)

		self.layerb11 = nn.Linear(1*1*512, self.branch_size, bias = False)
		self.layerb12 = nn.BatchNorm1d(self.branch_size)
		self.layerb13 = nn.Dropout(0.5)
		self.layerb14 = nn.Linear(self.branch_size, self.branch_size, bias = False)
		self.layerb15 = nn.BatchNorm1d(self.branch_size)
		self.layerb16 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(self.branch_size, self.branch_size)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)


	def forward_conv(self, z):
		
		z = self.layer11(z)
		z = self.activation(z)
		z = self.layer12(z)
		z = self.layer13(z)
		z = self.activation(z)
		z = self.layer14(z)
		z = self.layer15(z)
		z = self.activation(z)
		z = self.layer16(z)
		z = self.layer17(z)
		z = self.activation(z)
		z = self.layer18(z)
		z = self.layer19(z)
		
		z = self.layer20(z)
		z = self.activation(z)
		z = self.layer21(z)
		z = self.layer22(z)
		z = self.activation(z)
		z = self.layer23(z)
		z = self.layer24(z)
		z = self.activation(z)
		z = self.layer25(z)
		z = self.layer26(z)
		z = self.activation(z)
		z = self.layer27(z)
		z = self.layer28(z)
		
		z = self.layer29(z)
		z = self.activation(z)
		z = self.layer30(z)
		z = self.layer31(z)
		z = self.activation(z)
		z = self.layer32(z)
		z = self.layer33(z)
		z = self.activation(z)
		z = self.layer34(z)
		z = self.layer35(z)
		z = self.activation(z)
		z = self.layer36(z)
		
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
		return "HVGG_c3_b2_19"
