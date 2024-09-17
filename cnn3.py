import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
#from torchviz import make_dot
from torch.linalg import vector_norm as vnorm
from torch.linalg import solve as solve_matrix_system

from abc import ABC
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 


num_cores = 36
torch.set_num_interop_threads(num_cores) # Inter-op parallelism
torch.set_num_threads(num_cores) # Intra-op parallelism
device = "cuda:0"
#device = "cpu"
		
		

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
		#P = solve_matrix_system(WWT + torch.randn_like(WWT, device = device) * eps, I_o) # Broadcasting
		P = solve_matrix_system(WWT, I_o) # Broadcasting
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
			
		if loss_i1.isnan() == False:
			loss_i1.backward(retain_graph=True)
			back = True
			
		if loss_i2.isnan() == False:
			loss_i2.backward()
			back = True

		if back:
			self.optimizer.step()
		
		return torch.tensor([loss_f_vect.mean(), loss_i1_vect.mean(), loss_i2_vect.mean()]).clone().detach(), \
					torch.tensor([torch.heaviside(self.ort2-1e-7, self.v).sum(dim=1).mean(), torch.heaviside(self.ort3-1e-7, self.v).sum(dim=1).mean()]).clone().detach()
	

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
				iter = 0

			iter +=1

	
	def train_model(self, track = False, filename = ""):
		if not self.custom_training:
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
		str_bot_short = ""
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
		
		str_bot_short += f'{(100 * self.correct_c1_pred.sum() / self.total_c1_pred.sum()):.2f} %'
		str_bot_short += '  '

		str += f'Accuracy on c2: {(100 * self.correct_c2_pred.sum() / self.total_c2_pred.sum()):.2f} %'
		str += '\n'

		str_bot += f'Accuracy on c2: {(100 * self.correct_c2_pred.sum() / self.total_c2_pred.sum()):.2f} %'
		str_bot += '\n'
		
		str_bot_short += f'{(100 * self.correct_c2_pred.sum() / self.total_c2_pred.sum()):.2f} %'
		str_bot_short += '  '

		str += f'Accuracy on c3: {(100 * self.correct_c3_pred.sum() / self.total_c3_pred.sum()):.2f} %'
		str += '\n'

		str_bot += f'Accuracy on c3: {(100 * self.correct_c3_pred.sum() / self.total_c3_pred.sum()):.2f} %'
		str_bot += '\n'
		
		str_bot_short += f'{(100 * self.correct_c3_pred.sum() / self.total_c3_pred.sum()):.2f} %'
		str_bot_short += '\n'
		
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
		
		str_bot_short += f'{100 * float(self.correct_c1_vs_c2_pred.sum()) / self.total_c1_vs_c2_pred.sum():.2f} %'
		str_bot_short += '  '
		str_bot_short += f'{100 * float(self.correct_c2_vs_c3_pred.sum()) / self.total_c2_vs_c3_pred.sum():.2f} %'
		str_bot_short += '  '
		str_bot_short += f'{100 * float(self.correct_c1_vs_c3_pred.sum()) / self.total_c1_vs_c3_pred.sum():.2f} %'
		str_bot_short += '  '

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
			

		return str, str_bot, str_bot_short
		

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
				msg, msg_bot, msg_bot_short = self.test_results_to_text()
				with open(filename+"_test_performance.txt", 'w') as f:
					f.write(msg)
				return msg_bot, msg_bot_short

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
		msg += "Branch size: " +  str(self.branch_size) + "\n\n"
		msg += "Loss threshold: " + str(self.threshold) + "\n\n"
		msg += "Only thresholded: " + str(self.only_thresholded) + "\n\n"
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
		return self.dataset.c2_reinforce(c1_logits - torch.max(c1_logits), self.c2_reinforcer)
		

	def c3_reinforce(self, c2_logits):
		return self.dataset.c3_reinforce(c2_logits - torch.max(c2_logits), self.c3_reinforcer)
