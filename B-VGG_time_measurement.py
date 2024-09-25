import cnn3
from cnn3 import CNN3
from cnn3 import torch, optim, nn, F
from fashion_mnist import FashionMnist
from cnn3 import np
from cnn3 import device
import sys
from timeit import default_timer as timer

from telegramBot import Terminator

class BVGG(CNN3):
	def __init__(self, alpha, beta, gamma, weights_switch_points, learning_rate, lr_switch_points, momentum, nesterov, dataset, epochs, every_print = 512):
		
		super().__init__(learning_rate, momentum, nesterov, dataset, epochs, every_print, custom_training = True)
		
		self.alphas = alpha
		self.betas = beta
		self.gammas = gamma
		self.alpha = self.alphas[0]
		self.beta = self.betas[0]
		self.gamma = self.gammas[0] 
		
		self.weights_switch_points = weights_switch_points
		self.lr_switch_points = lr_switch_points

		# Block 1
		self.layer1  = nn.Conv2d(1, 64, (3,3), padding = 'same')
		self.layer2  = nn.BatchNorm2d(64)
		self.layer3  = nn.Conv2d(64, 64, (3,3), padding = 'same')
		self.layer4  = nn.BatchNorm2d(64)
		self.layer5  = nn.MaxPool2d((2,2), stride = (2,2))


		# Block 2
		self.layer6  = nn.Conv2d(64, 128, (3,3), padding = 'same')
		self.layer7  = nn.BatchNorm2d(128)
		self.layer8  = nn.Conv2d(128, 128, (3,3), padding = 'same')
		self.layer9  = nn.BatchNorm2d(128)
		self.layer10 = nn.MaxPool2d((2,2), stride = (2,2))


		# Coarse 1
		self.layerb11 = nn.Linear(7*7*128, 256)
		self.layerb12 = nn.BatchNorm1d(256)
		self.layerb13 = nn.Dropout(0.5)
		self.layerb14 = nn.Linear(256, 256)
		self.layerb15 = nn.BatchNorm1d(256)
		self.layerb16 = nn.Dropout(0.5)
		self.layerb17 = nn.Linear(256, self.dataset.num_c1)
		
		
		# Coarse 2
		self.layerb21 = nn.Linear(3*3*256, 1024)
		self.layerb22 = nn.BatchNorm1d(1024)
		self.layerb23 = nn.Dropout(0.5)
		self.layerb24 = nn.Linear(1024, 1024)
		self.layerb25 = nn.BatchNorm1d(1024)
		self.layerb26 = nn.Dropout(0.5)
		self.layerb27 = nn.Linear(1024, self.dataset.num_c2)

		
		# Coarse 3
		self.layerb31 = nn.Linear(1*1*512, 4096)
		self.layerb32 = nn.BatchNorm1d(4096)
		self.layerb33 = nn.Dropout(0.5)
		self.layerb34 = nn.Linear(4096, 4096)
		self.layerb35 = nn.BatchNorm1d(4096)
		self.layerb36 = nn.Dropout(0.5)
		self.layerb37 = nn.Linear(4096, self.dataset.num_c3)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss()


	def forward(self, x):
		pass


	def predict_and_learn_naive(self, batch, labels):
		self.optimizer.zero_grad()
		predict = self(batch)
		
		loss1 = self.criterion(predict[0], labels[:,0])
		loss2 = self.criterion(predict[1], labels[:,1])
		loss3 = self.criterion(predict[2], labels[:,2])
		
		loss =  self.alpha * loss1 + self.beta * loss2 + self.gamma * loss3

		loss.backward()
		self.optimizer.step()

		return torch.tensor([loss1, loss2, loss3]).clone().detach(), loss.clone().detach()

	
	def training_loop_body_track(self):
		running_loss = torch.zeros(self.dataset.class_levels)
		running_loss_scalar = 0.0
		
		iter = 1
	
		for batch, labels in self.dataset.trainloader:
			batch = batch.to(device)
			labels = labels.to(device)
		
			loss, loss_scalar = self.predict_and_learn_(batch, labels)

			running_loss += (loss - running_loss) / iter
			running_loss_scalar += (loss_scalar - running_loss_scalar) / iter
			
			if iter & self.every_print == 0:
				self.loss_track[self.num_push, :] = running_loss
				self.loss_track_scalar[self.num_push] = running_loss_scalar
				self.accuracy_track[self.num_push, :] = self.test(mode = "train")
				self.num_push += 1
				running_loss = torch.zeros(self.dataset.class_levels)
				running_loss_scalar = 0.0
				iter = 0

			iter +=1


	def custom_training_f(self, track = False, filename = ""):
	
		training_loop_f = self.training_loop_body
		
		if track:
			training_loop_f = self.training_loop_body_track
			self.loss_track_scalar = torch.zeros(self.epochs * self.track_size)

		for epoch in np.arange(self.weights_switch_points[0]):
			training_loop_f()
			
		self.alpha = self.alphas[1]
		self.beta = self.betas[1]
		self.gamma = self.gammas[1]
			
		for epoch in np.arange(self.weights_switch_points[0], self.weights_switch_points[1]):
			training_loop_f()
			
		self.alpha = self.alphas[2]
		self.beta = self.betas[2]
		self.gamma = self.gammas[2]

		for epoch in np.arange(self.weights_switch_points[1], self.weights_switch_points[2]):
			training_loop_f()
		
		self.alpha = self.alphas[3]
		self.beta = self.betas[3]
		self.gamma = self.gammas[3]
		
		for epoch in np.arange(self.weights_switch_points[2], self.lr_switch_points[0]):
			training_loop_f()
			
		self.optimizer.param_groups[0]['lr'] = self.learning_rate[1]	
			
		for epoch in np.arange(self.lr_switch_points[0], self.lr_switch_points[1]):
			training_loop_f()
			
		self.optimizer.param_groups[0]['lr'] = self.learning_rate[2]

		for epoch in np.arange(self.lr_switch_points[1], self.epochs):
			training_loop_f()
		
		if track:
			self.plot_training_loss_scalar(filename + "_train_loss_scalar.pdf")

		
	def plot_training_loss_scalar(self, filename):
		plt.figure(figsize=(12, 6))
		plt.plot(np.linspace(1, self.epochs, self.loss_track_scalar.size(0)), self.loss_track_scalar.numpy())
		plt.title("Weighted training loss")
		plt.xlabel("Epochs")
		plt.ylabel("Error")
		plt.xticks(np.linspace(1, self.epochs, self.epochs)[0::2])
		plt.savefig(filename, bbox_inches='tight')
		plt.close()


	def __str__(self):
		pass
		

class BVGG19(BVGG):
	def __init__(self, alpha, beta, gamma, weights_switch_points, learning_rate, lr_switch_points, momentum, nesterov, dataset, epochs, every_print = 512):

		super().__init__(alpha, beta, gamma, weights_switch_points, learning_rate, lr_switch_points, momentum, nesterov, dataset, epochs, every_print)


		# Block 3
		self.layer11 = nn.Conv2d(128, 256, (3,3), padding = 'same')
		self.layer12 = nn.BatchNorm2d(256)
		self.layer13 = nn.Conv2d(256, 256, (3,3), padding = 'same')
		self.layer14 = nn.BatchNorm2d(256)
		self.layer15 = nn.Conv2d(256, 256, (3,3), padding = 'same')
		self.layer16 = nn.BatchNorm2d(256)
		self.layer17 = nn.Conv2d(256, 256, (3,3), padding = 'same')
		self.layer18 = nn.BatchNorm2d(256)
		self.layer19 = nn.MaxPool2d((2,2), stride = (2,2))


		# Block 4
		self.layer20 = nn.Conv2d(256, 512, (3,3), padding = 'same')
		self.layer21 = nn.BatchNorm2d(512)
		self.layer22 = nn.Conv2d(512, 512, (3,3), padding = 'same')
		self.layer23 = nn.BatchNorm2d(512)
		self.layer24 = nn.Conv2d(512, 512, (3,3), padding = 'same')
		self.layer25 = nn.BatchNorm2d(512)
		self.layer26 = nn.Conv2d(512, 512, (3,3), padding = 'same')
		self.layer27 = nn.BatchNorm2d(512)
		self.layer28 = nn.MaxPool2d((2,2), stride = (2,2))

		
		# Block 5
		self.layer29 = nn.Conv2d(512, 512, (3,3), padding = 'same')
		self.layer30 = nn.BatchNorm2d(512)
		self.layer31 = nn.Conv2d(512, 512, (3,3), padding = 'same')
		self.layer32 = nn.BatchNorm2d(512)
		self.layer33 = nn.Conv2d(512, 512, (3,3), padding = 'same')
		self.layer34 = nn.BatchNorm2d(512)
		self.layer35 = nn.Conv2d(512, 512, (3,3), padding = 'same')
		self.layer36 = nn.BatchNorm2d(512)


		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss()


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

		# branch 1
		b1 = torch.flatten(z, start_dim = 1)
		b1 = self.layerb11(b1)
		b1 = self.activation(b1)
		b1 = self.layerb12(b1)
		b1 = self.layerb13(b1)
		b1 = self.layerb14(b1)
		b1 = self.activation(b1)
		b1 = self.layerb15(b1)
		b1 = self.layerb16(b1)
		b1 = self.layerb17(b1)

		# block 3
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
		
		# branch 2
		b2 = torch.flatten(z, start_dim = 1)
		b2 = self.layerb21(b2)
		b2 = self.activation(b2)
		b2 = self.layerb22(b2)
		b2 = self.layerb23(b2)
		b2 = self.layerb24(b2)
		b2 = self.activation(b2)
		b2 = self.layerb25(b2)
		b2 = self.layerb26(b2)
		b2 = self.layerb27(b2)

		# block 4
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


		# block 5
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

		# branch 3
		b3 = torch.flatten(z, start_dim = 1)
		b3 = self.layerb31(b3)
		b3 = self.activation(b3)
		b3 = self.layerb32(b3)
		b3 = self.layerb33(b3)
		b3 = self.layerb34(b3)
		b3 = self.activation(b3)
		b3 = self.layerb35(b3)
		b3 = self.layerb36(b3)
		b3 = self.layerb37(b3)
		
		
		return b1, b2, b3
		


	def __str__(self):
		return "B-VGG16"

class BVGG16(BVGG):
	def __init__(self, alpha, beta, gamma, weights_switch_points, learning_rate, lr_switch_points, momentum, nesterov, dataset, epochs, every_print = 512):
		
		super().__init__(alpha, beta, gamma, weights_switch_points, learning_rate, lr_switch_points, momentum, nesterov, dataset, epochs, every_print)


		# Block 3
		self.layer11 = nn.Conv2d(128, 256, (3,3), padding = 'same')
		self.layer12 = nn.BatchNorm2d(256)
		self.layer13 = nn.Conv2d(256, 256, (3,3), padding = 'same')
		self.layer14 = nn.BatchNorm2d(256)
		self.layer15 = nn.Conv2d(256, 256, (3,3), padding = 'same')
		self.layer16 = nn.BatchNorm2d(256)
		self.layer17 = nn.MaxPool2d((2,2), stride = (2,2))


		# Block 4
		self.layer18 = nn.Conv2d(256, 512, (3,3), padding = 'same')
		self.layer19 = nn.BatchNorm2d(512)
		self.layer20 = nn.Conv2d(512, 512, (3,3), padding = 'same')
		self.layer21 = nn.BatchNorm2d(512)
		self.layer22 = nn.Conv2d(512, 512, (3,3), padding = 'same')
		self.layer23 = nn.BatchNorm2d(512)
		self.layer24 = nn.MaxPool2d((2,2), stride = (2,2))

		
		# Block 5
		self.layer25 = nn.Conv2d(512, 512, (3,3), padding = 'same')
		self.layer26 = nn.BatchNorm2d(512)
		self.layer27 = nn.Conv2d(512, 512, (3,3), padding = 'same')
		self.layer28 = nn.BatchNorm2d(512)
		self.layer29 = nn.Conv2d(512, 512, (3,3), padding = 'same')
		self.layer30 = nn.BatchNorm2d(512)

		self.optimizer = optim.SGD(self.parameters(), lr = self.learning_rate[0], momentum = self.momentum, nesterov = self.nesterov)
		self.criterion = nn.CrossEntropyLoss()


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

		# branch 1
		b1 = torch.flatten(z, start_dim = 1)
		b1 = self.layerb11(b1)
		b1 = self.activation(b1)
		b1 = self.layerb12(b1)
		b1 = self.layerb13(b1)
		b1 = self.layerb14(b1)
		b1 = self.activation(b1)
		b1 = self.layerb15(b1)
		b1 = self.layerb16(b1)
		b1 = self.layerb17(b1)

		# block 3
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
		
		# branch 2
		b2 = torch.flatten(z, start_dim = 1)
		b2 = self.layerb21(b2)
		b2 = self.activation(b2)
		b2 = self.layerb22(b2)
		b2 = self.layerb23(b2)
		b2 = self.layerb24(b2)
		b2 = self.activation(b2)
		b2 = self.layerb25(b2)
		b2 = self.layerb26(b2)
		b2 = self.layerb27(b2)

		# block 4
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


		# block 5
		z = self.layer25(z)
		z = self.activation(z)
		z = self.layer26(z)
		z = self.layer27(z)
		z = self.activation(z)
		z = self.layer28(z)
		z = self.layer29(z)
		z = self.activation(z)
		z = self.layer30(z)

		# branch 3
		b3 = torch.flatten(z, start_dim = 1)
		b3 = self.layerb31(b3)
		b3 = self.activation(b3)
		b3 = self.layerb32(b3)
		b3 = self.layerb33(b3)
		b3 = self.layerb34(b3)
		b3 = self.activation(b3)
		b3 = self.layerb35(b3)
		b3 = self.layerb36(b3)
		b3 = self.layerb37(b3)
		
		
		return b1, b2, b3
		

	def __str__(self):
		return "B-VGG16"

			
if __name__ == '__main__':

	cnn3.torch.autograd.set_detect_anomaly(False);
	cnn3.torch.autograd.profiler.emit_nvtx(False);
	cnn3.torch.autograd.profiler.profile(False);

	alpha = [0.98, 0.1, 0.1, 0.]
	beta = [0.01, 0.8, 0.2, 0.]
	gamma = [0.01, 0.1, 0.7, 1.]
	momentum = 0.9
	nesterov = True
	every_print = 32
	batch_size = 128
	track = False
	
	if sys.argv[1] == "CIFAR100":
		learning_rate = [1e-3, 2e-4, 5e-5]
		epochs = 80
		weights_switch_points = [13, 23, 33]
		lr_switch_points = [56, 71]
		dataset = CIFAR100(batch_size)
		
	elif sys.argv[1] == "CIFAR10":
		learning_rate = [3e-3, 5e-4, 1e-4]
		epochs = 60
		weights_switch_points = [10, 20, 30]
		lr_switch_points = [43, 53]
		dataset = CIFAR10(batch_size)
		
	elif sys.argv[1] == "FASHION":
		learning_rate = [1e-3, 1e-4, 5e-5]
		epochs = 80
		weights_switch_points = [15, 25, 35]
		lr_switch_points = [42, 52]
		dataset = FashionMnist(batch_size)
		
	else:
		raise ValueError(f'Dataset {sys.argv[1]} is not supported yet.')
	
	bot = Terminator()
	
	if sys.argv[2] == "16":
		cnn = BVGG16(alpha, beta, gamma, weights_switch_points, learning_rate, lr_switch_points, momentum, nesterov, dataset, epochs, every_print)
		
	elif sys.argv[2] == "19":
		cnn = BVGG19(alpha, beta, gamma, weights_switch_points, learning_rate, lr_switch_points, momentum, nesterov, dataset, epochs, every_print)
		
	else:
		raise ValueError(f'Model {sys.argv[2]} is not supported yet.')
		
	cnn.to(device)
	
	err = False
	filename = "models/" + str(dataset) + "/B-VGG/time.txt"

	try:
		start = timer()

		cnn.train_model(track, filename)

		end = timer()

		time_spent = end - start
		msg = "time total: " + str(time_spent)

		time_spent_per_epoch = time_spent / epochs
		msg += "\n\ntime per epoch: " + str(time_spent_per_epoch)

		time_spent_per_batch = time_spent_per_epoch / cnn.dataset.training_size * cnn.dataset.batch_size
		msg += "\n\ntime per batch: " + str(time_spent_per_batch)

		bot.sendMessage("Test completato correttamente")
		bot.sendMessage("Tempo impiegato: " + str(time_spent) + "\n\n" + "Tempo impiegato per epoca: " + str(time_spent_per_epoch) + "\n\n" + "Tempo impiegato per batch: " + str(time_spent_per_batch))

		with open(filename, "w") as f:
			f.write(msg)
		
	except Exception as errore:
		err = errore

	if err is False:
		bot.sendMessage("Programma terminato correttamente\n\n\nPerformance:\n\n"+msg)
	else:
		bot.sendMessage("Programma NON terminato correttamente\nTipo di errore: "+err.__class__.__name__+"\nMessaggio: "+str(err))
		raise err
