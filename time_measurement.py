from telegramBot import Terminator
import HCNN3
from HCNN3 import HCNN3_c2_b0
from HCNN3 import HCNN3_c4_b0
from cnn3 import CIFAR100, CIFAR10
from cnn3 import device
import sys
from timeit import default_timer as timer


class Configuration:
	
	def __init__(self, learning_rate, epochs, switch_points, momentum, nesterov, 
					every_print, custom_training, threshold, reduction, track, dataset, model, branch_size, reinforce, projection, only_thresholded):
		
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.nesterov = nesterov
		self.epochs = epochs
		self.switch_points = switch_points
		self.every_print = every_print
		self.custom_training = custom_training
		self.threshold = threshold
		self.reduction = reduction
		self.track = track
		self.dataset = dataset
		self.model = model
		self.branch_size = branch_size
		self.reinforce = reinforce
		self.projection = projection
		self.only_thresholded = only_thresholded
		
		
	def __str__(self):
		msg = ""
		msg += "Epochs: " + str(self.epochs) + "\n"
		msg += "LR: " 
		for lr in self.learning_rate:
			msg += str(lr) + " "
		msg += "\n"
		msg += "Switch point: "
		for sp in self.switch_points:
			msg += str(sp) + " "
		msg += "\n"
		msg += "Branch size: " + str(self.branch_size) +  "\n"
		msg += "Loss threshold: " + str(self.threshold) + "\n"
		msg += "Only thresholded: " + str(self.only_thresholded) + "\n"
		msg += "Reduction: " + self.reduction + "\n"
		msg += "Reinforce: " + str(self.reinforce) + "\n"
		msg += "Projection: " + str(self.projection)
		
		return msg


class Tester():

	def __init__(self, conf):
	
		self.bot = Terminator()
		self.conf = conf
		
	
	def launch(self):
	
		try:

			cnn = self.conf.model(self.conf.learning_rate, self.conf.momentum, self.conf.nesterov, self.conf.dataset, 
					2, self.conf.every_print, self.conf.switch_points, self.conf.custom_training, self.conf.threshold, 
					self.conf.reduction, self.conf.branch_size, self.conf.reinforce, self.conf.projection, self.conf.only_thresholded)
					
			
			filename = "models/" + str(conf.dataset) + "/" + str(cnn) + "_time"
			
			cnn.to(device)

			cnn.train_model(False)

			cnn.epochs = self.conf.epochs - 2
			
			start = timer()

			cnn.train_model(False)

			end = timer()
			
			additional_info = ""
			if conf.reinforce:
				additional_info += "reinforce\t"
			if conf.projection:
				additional_info += "projection\t"

			time_spent = end - start
			additional_info += "\n\ntime total: " + str(time_spent)

			time_spent_per_epoch = time_spent / self.conf.epochs
			additional_info += "\n\ntime per epoch: " + str(time_spent_per_epoch)

			time_spent_per_batch = time_spent_per_epoch / cnn.dataset.training_size * cnn.dataset.batch_size
			additional_info += "\n\ntime per batch: " + str(time_spent_per_batch)
				
			cnn.write_configuration(filename, additional_info)
					
		except Exception as error:
			self.bot.sendMessage("Programma NON terminato correttamente\nTipo di errore: " + error.__class__.__name__ + "\nMessaggio: " + str(error))
			raise error
			
		self.bot.sendMessage("Test completato correttamente")
		self.bot.sendMessage("Tempo impiegato: " + str(time_spent) + "\n\n" + "Tempo impiegato per epoca: " + str(time_spent_per_epoch) + "\n\n" + "Tempo impiegato per batch: " + str(time_spent_per_batch))


if __name__ == '__main__':

	HCNN3.cnn3.torch.autograd.set_detect_anomaly(False);
	HCNN3.cnn3.torch.autograd.profiler.emit_nvtx(False);
	HCNN3.cnn3.torch.autograd.profiler.profile(False);
	
	# fixed params
	batch_size = 128
	track = False
	custom_training = False
	momentum = 0.9
	nesterov = True
	every_print = 32
	reinforce = True
	projection = False
	only_thresholded = False
	threshold = 0.0
	reduction = 'mean'
	model = HCNN3_c4_b0

	if sys.argv[1] == "CIFAR100":
		# variable params
		dataset = CIFAR100(batch_size)
		lr = [1e-3, 2e-4]
		branch_size = 512

		conf = Configuration(lr, 15, [11], momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, model, branch_size, reinforce, projection, only_thresholded)
		

	elif sys.argv[1] == "CIFAR10":
	
		dataset = CIFAR10(batch_size)
		lr = [3e-3, 5e-4]
		branch_size = 256

		conf = Configuration(lr, 20, [11], momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, model, branch_size, reinforce, projection, only_thresholded)
		

	else:
		raise ValueError(f'Test for {sys.argv[1]} is not supported yet.')


	
	t = Tester(conf)
	
	t.launch()
