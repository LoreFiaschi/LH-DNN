from telegramBot import Terminator
import HCNN3
from HCNN3 import HCNN3_c0_b0_r, HCNN3_c0_b1_r, HCNN3_c0_b2_r, HCNN3_c1_b0_r, HCNN3_c1_b1_r, HCNN3_c1_b2_r, HCNN3_c2_b0_r, HCNN3_c2_b1_r, HCNN3_c2_b2_r
from cnn3 import CIFAR100, CIFAR10
from cnn3 import device
from tqdm import tqdm 


class Configuration:
	
	def __init__(self, learning_rate, epochs, switch_point, batch_size, momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size):
		
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.nesterov = nesterov
		self.epochs = epochs
		self.switch_point = switch_point
		self.every_print = every_print
		self.custom_training = custom_training
		self.threshold = threshold
		self.reduction = reduction
		self.track = track
		self.dataset = dataset
		self.batch_size = batch_size
		self.models = list_of_models
		self.branch_size = branch_size


class Tester():

	def __init__(self, list_of_conf):
	
		self.bot = Terminator()
		self.list_of_conf = list_of_conf
		
	
	def launch(self):
	
		try:
			for num, conf in tqdm(enumerate(self.list_of_conf), desc = "Testing configuration"):
			
				dataset = conf.dataset(conf.batch_size)
				
				for model in conf.models:
					cnn = model(conf.learning_rate, conf.momentum, conf.nesterov, dataset, 
							conf.epochs, conf.every_print, conf.switch_point, conf.custom_training, conf.threshold, conf.reduction, conf.branch_size)
							
					
					filename = "models/" + str(dataset) + "/" + str(cnn) + "_conf_" + str(num)
					
					cnn.to(device)
					
					cnn.train_model(conf.track, filename)
					cnn.save_model(filename)
					msg = cnn.test(mode = "write", filename = filename)
					cnn.write_configuration(filename, "reinforce\n\n" + msg)
					
		except Exception as error:
			self.bot.sendMessage("Programma NON terminato correttamente\nTipo di errore: " + error.__class__.__name__ + "\nMessaggio: " + str(error))
			raise error


		self.bot.sendMessage("Test completato correttamente")


if __name__ == '__main__':

	HCNN3.cnn3.torch.autograd.set_detect_anomaly(False);
	HCNN3.cnn3.torch.autograd.profiler.emit_nvtx(False);
	HCNN3.cnn3.torch.autograd.profiler.profile(False);
	
	# fixed params
	batch_size = 128
	track = True
	custom_training = False
	threshold = 0.6
	reduction = 'none'
	momentum = 0.9
	nesterov = True
	every_print = 32
	
	"""
	
	dataset = CIFAR100
	lr2 = [1e-3, 2e-4]
	lr3 = [1e-3, 2e-4, 5e-5]
	branch_size = 512
	
	"""
	
	list_of_models = [HCNN3_c0_b0_r, HCNN3_c0_b1_r, HCNN3_c0_b2_r, HCNN3_c1_b0_r, HCNN3_c1_b1_r, HCNN3_c1_b2_r, HCNN3_c2_b0_r, HCNN3_c2_b1_r, HCNN3_c2_b2_r]
	
	list_of_conf = []

	"""

	# changing params: learning_rate, epochs, switch_point
	
	list_of_conf.append(Configuration(lr2, 9, [5], batch_size, momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size))
	list_of_conf.append(Configuration(lr2, 11, [7], batch_size, momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size))
	list_of_conf.append(Configuration(lr2, 15, [9], batch_size, momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size))
	list_of_conf.append(Configuration(lr2, 15, [11], batch_size, momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size))
	list_of_conf.append(Configuration(lr2, 15, [7], batch_size, momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size))
	list_of_conf.append(Configuration(lr2, 20, [11], batch_size, momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size))
	list_of_conf.append(Configuration(lr2, 20, [12], batch_size, momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size))
	list_of_conf.append(Configuration(lr3, 15, [9, 13], batch_size, momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size))
	list_of_conf.append(Configuration(lr3, 15, [11, 13], batch_size, momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size))
	list_of_conf.append(Configuration(lr3, 17, [11, 14], batch_size, momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size))
	list_of_conf.append(Configuration(lr3, 20, [11, 14], batch_size, momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size))
	
	"""
	
	dataset = CIFAR10
	lr = [3e-3, 5e-4]
	branch_size = 128
	
	list_of_conf.append(Configuration(lr, 9, [5], batch_size, momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size))
	list_of_conf.append(Configuration(lr, 11, [5], batch_size, momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size))
	list_of_conf.append(Configuration(lr, 11, [7], batch_size, momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size))
	list_of_conf.append(Configuration(lr, 11, [9], batch_size, momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size))
	list_of_conf.append(Configuration(lr, 15, [9], batch_size, momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size))
	list_of_conf.append(Configuration(lr, 15, [11], batch_size, momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size))
	list_of_conf.append(Configuration(lr, 20, [9], batch_size, momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size))
	list_of_conf.append(Configuration(lr, 20, [11], batch_size, momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size))
	
	t = Tester(list_of_conf)
	
	t.launch()
