import cnn3

from cnn3 import torch, optim
from cnn3 import CNN3
from cnn3 import nn
from cnn3 import np


class HCNN3(CNN3):
	
	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction):
		
		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction)
		
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
		
		self.layerb17 = nn.Linear(512, self.dataset.num_c1)
		
		self.layerb27 = nn.Linear(512, self.dataset.num_c2)
		self.layerb27__ = nn.Linear(512, self.dataset.num_c2)
		
		self.layerb37 = nn.Linear(512, self.dataset.num_c3)
		self.layerb37__ = nn.Linear(512, self.dataset.num_c3)
	
	
	def forward_conv(self, x):
		return x
		
		
	def forward_branch(self, x):
		return x
	
		
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
		b2 = self.layerb27(ort2) + self.layerb27__(prj2) + self.c2_reinforce(b1.clone().detach())

		# branch 3
		b3 = self.layerb37(ort3) + self.layerb37__(prj3) + self.c3_reinforce(b2.clone().detach())

		return b1, b2, b3
		

class HCNN3_c0_b0_r(HCNN3):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean'):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction)

		self.layerb_mid = nn.Linear(8*8*128, 512)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss(reduction = reduction)
			
	
	def forward_branch(self, z):
	
		z = self.layerb_mid(z)
		z = self.activation(z)
		
		return z
		
		
	def __str__(self):
		return "HCNN3_c0_b0_r"



class HCNN3_c0_b1_r(HCNN3):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean'):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction)

		self.layerb11 = nn.Linear(8*8*128, 512, bias = False)
		self.layerb12 = nn.BatchNorm1d(512)
		self.layerb13 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(512, 512)

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
		return "HCNN3_c0_b1_r"
		

		
class HCNN3_c0_b2_r(HCNN3):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean'):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction)

		self.layerb11 = nn.Linear(8*8*128, 512, bias = False)
		self.layerb12 = nn.BatchNorm1d(512)
		self.layerb13 = nn.Dropout(0.5)
		self.layerb14 = nn.Linear(512, 512, bias = False)
		self.layerb15 = nn.BatchNorm1d(512)
		self.layerb16 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(512, 512)

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
		return "HCNN3_c0_b2_r"
		
		
		
class HCNN3_c1_b0_r(HCNN3):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean'):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction)
		
		self.layer10_1 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer10_2 = nn.BatchNorm2d(256)
		self.layer10_5 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb_mid = nn.Linear(8*8*128, 512)

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
		return "HCNN3_c1_b0_r"
		
		

class HCNN3_c1_b1_r(HCNN3):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean'):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction)
		
		self.layer10_1 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer10_2 = nn.BatchNorm2d(256)
		self.layer10_5 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb11 = nn.Linear(8*8*128, 512, bias = False)
		self.layerb12 = nn.BatchNorm1d(512)
		self.layerb13 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(512, 512)

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
		return "HCNN3_c1_b1_r"
		
		
		
class HCNN3_c1_b2_r(HCNN3):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean'):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction)
		
		self.layer10_1 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer10_2 = nn.BatchNorm2d(256)
		self.layer10_5 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb11 = nn.Linear(8*8*128, 512, bias = False)
		self.layerb12 = nn.BatchNorm1d(512)
		self.layerb13 = nn.Dropout(0.5)
		self.layerb14 = nn.Linear(512, 512, bias = False)
		self.layerb15 = nn.BatchNorm1d(512)
		self.layerb16 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(512, 512)

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
		return "HCNN3_c1_b2_r"	



class HCNN3_c2_b0_r(HCNN3):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean'):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction)
		
		self.layer10_1 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer10_2 = nn.BatchNorm2d(256)
		self.layer10_3 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer10_4 = nn.BatchNorm2d(256)
		self.layer10_5 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb_mid = nn.Linear(8*8*128, 512)

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
		return "HCNN3_c2_b0_r"	



class HCNN3_c2_b1_r(HCNN3):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean'):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction)
		
		self.layer10_1 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer10_2 = nn.BatchNorm2d(256)
		self.layer10_3 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer10_4 = nn.BatchNorm2d(256)
		self.layer10_5 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb11 = nn.Linear(8*8*128, 512, bias = False)
		self.layerb12 = nn.BatchNorm1d(512)
		self.layerb13 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(512, 512)

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
		return "HCNN3_c2_b1_r"	
	
		
		
class HCNN3_c2_b2_r(HCNN3):

	def __init__(self, learning_rate, momentum, nesterov, dataset, epochs, every_print = 512, switch_point = None, custom_training = False, threshold = 0.1, reduction = 'mean'):

		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, switch_point, custom_training, threshold, reduction)
		
		self.layer10_1 = nn.Conv2d(128, 256, (3,3), padding = 'same', bias = False)
		self.layer10_2 = nn.BatchNorm2d(256)
		self.layer10_3 = nn.Conv2d(256, 256, (3,3), padding = 'same', bias = False)
		self.layer10_4 = nn.BatchNorm2d(256)
		self.layer10_5 = nn.MaxPool2d((2,2), stride = (2,2))

		self.layerb11 = nn.Linear(8*8*128, 512, bias = False)
		self.layerb12 = nn.BatchNorm1d(512)
		self.layerb13 = nn.Dropout(0.5)
		self.layerb14 = nn.Linear(512, 512, bias = False)
		self.layerb15 = nn.BatchNorm1d(512)
		self.layerb16 = nn.Dropout(0.5)

		self.layerb_mid = nn.Linear(512, 512)
		
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
		return "HCNN3_c2_b2_r"
