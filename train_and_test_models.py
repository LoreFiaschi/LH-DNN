from telegramBot import Terminator
import HCNN3
from HCNN3 import HCNN3_c0_b0, HCNN3_c0_b1, HCNN3_c0_b2
from HCNN3 import HCNN3_c1_b0, HCNN3_c1_b1, HCNN3_c1_b2
from HCNN3 import HCNN3_c2_b0, HCNN3_c2_b1, HCNN3_c2_b2
from HCNN3 import HCNN3_c3_b0, HCNN3_c3_b1, HCNN3_c3_b2
from HCNN3 import HCNN3_c4_b0, HCNN3_c4_b1, HCNN3_c4_b2
from cnn3 import CIFAR100, CIFAR10
from cnn3 import device
from tqdm import tqdm 
import sys


class Configuration:
	
	def __init__(self, learning_rate, epochs, switch_points, momentum, nesterov, 
					every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size, reinforce, projection, only_thresholded):
		
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
		self.models = list_of_models
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

	def __init__(self, list_of_conf):
	
		self.bot = Terminator()
		self.list_of_conf = list_of_conf
		
	
	def launch(self):
	
		resume = ""
		resume_short = ""
	
		try:
			
			num_conf = len(self.list_of_conf)
			
			for num in tqdm(range(num_conf), desc = "Testing configuration"):
			
				conf = self.list_of_conf[num]
				
				for model in conf.models:
					cnn = model(conf.learning_rate, conf.momentum, conf.nesterov, conf.dataset, 
							conf.epochs, conf.every_print, conf.switch_points, conf.custom_training, conf.threshold, conf.reduction, conf.branch_size, conf.reinforce, conf.projection, conf.only_thresholded)
							
					
					filename = "models/" + str(conf.dataset) + "/" + str(cnn) + "_conf_" + str(num)
					resume += '\t' + str(cnn) + "_conf_" + str(num) + '\n'
					resume_short += str(cnn) + "_conf_" + str(num) + ': '
					
					cnn.to(device)
					
					cnn.train_model(conf.track, filename)
					cnn.save_model(filename)
					msg, msg_short = cnn.test(mode = "write", filename = filename)
					
					additional_info = ""
					if conf.reinforce:
						additional_info += "reinforce\t"
					if conf.projection:
						additional_info += "projection\t"
						
					cnn.write_configuration(filename, additional_info + "\n\n" + msg)
					
					resume += msg + '\n\n'
					resume_short += msg_short + '\n'
					
			
			resume_filename = "models/" + str(conf.dataset) + "/" + "test_resume.txt"
					
			with open(resume_filename, 'w') as f:
				f.write(resume)
					
		except Exception as error:
			self.bot.sendMessage("Programma NON terminato correttamente\nTipo di errore: " + error.__class__.__name__ + "\nMessaggio: " + str(error))
			raise error
			
		self.bot.sendMessage("Test completato correttamente")
		self.bot.sendMessage(resume_short)
		
		
	def write_legend(self, filename):
		msg = ""
		for num_conf, conf in enumerate(self.list_of_conf):
			msg += f'\tConfiguration {num_conf}\n'
			msg += str(conf) + "\n\n"
			
		with open(filename, 'w') as f:
				f.write(msg)


if __name__ == '__main__':

	HCNN3.cnn3.torch.autograd.set_detect_anomaly(False);
	HCNN3.cnn3.torch.autograd.profiler.emit_nvtx(False);
	HCNN3.cnn3.torch.autograd.profiler.profile(False);
	
	# fixed params
	batch_size = 128
	track = True
	custom_training = False
	momentum = 0.9
	nesterov = True
	every_print = 32
	reinforce = True
	projection = False

	#c0_models = [HCNN3_c0_b0, HCNN3_c0_b1, HCNN3_c0_b2]
	#c1_models = [HCNN3_c1_b0, HCNN3_c1_b1, HCNN3_c1_b2]
	#c2_models = [HCNN3_c2_b0, HCNN3_c2_b1, HCNN3_c2_b2]
	#c3_models = [HCNN3_c3_b0, HCNN3_c3_b1, HCNN3_c3_b2]
	#c4_models = [HCNN3_c4_b0, HCNN3_c4_b1, HCNN3_c4_b2]
	
	b0_models = [HCNN3_c0_b0, HCNN3_c1_b0, HCNN3_c2_b0, HCNN3_c3_b0, HCNN3_c4_b0]
	b1_models = [HCNN3_c0_b1, HCNN3_c1_b1, HCNN3_c2_b1, HCNN3_c3_b1, HCNN3_c4_b1]
	b2_models = [HCNN3_c0_b2, HCNN3_c1_b2, HCNN3_c2_b2, HCNN3_c3_b2, HCNN3_c4_b2]
	
	list_of_conf = []
	
	if sys.argv[1] == "CIFAR100":

		dataset = CIFAR100(batch_size)
		lr2 = [1e-3, 2e-4]
		lr3 = [1e-3, 2e-4, 5e-5]
		only_thresholded = False
		threshold = 0.0
		reduction = 'mean'
		#list_of_models = c0_models + c2_models + c4_models
		list_of_models = [HCNN3_c2_b0] + [HCNN3_c4_b0]
		branch_size = 512

		#list_of_conf.append(Configuration(lr2, 9, [5], momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size, reinforce, projection, only_thresholded))
		#list_of_conf.append(Configuration(lr2, 11, [7], momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size, reinforce, projection, only_thresholded))
		#list_of_conf.append(Configuration(lr2, 15, [9], momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size, reinforce, projection, only_thresholded))
		list_of_conf.append(Configuration(lr2, 15, [11], momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size, reinforce, projection, only_thresholded))
		list_of_conf.append(Configuration(lr2, 20, [11], momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size, reinforce, projection, only_thresholded))
		#list_of_conf.append(Configuration(lr3, 15, [9, 11], momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size, reinforce, projection, only_thresholded))
		#list_of_conf.append(Configuration(lr3, 15, [11, 13], momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size, reinforce, projection, only_thresholded))
		#list_of_conf.append(Configuration(lr3, 20, [11, 14], momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size, reinforce, projection, only_thresholded))

	elif sys.argv[1] == "CIFAR10":
	
		dataset = CIFAR10(batch_size)
		lr = [3e-3, 5e-4]
		only_thresholded = False
		threshold = 0.6
		reduction = 'none'
		c02_b12_models = [b1_models[0], b1_models[2], b2_models[0], b2_models[2]]
		
		list_of_conf.append(Configuration(lr,  9, [5], momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, c02_b12_models, 256, reinforce, projection, only_thresholded))
		list_of_conf.append(Configuration(lr,  9, [5], momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, c4_models, 512, reinforce, projection, only_thresholded))
		list_of_conf.append(Configuration(lr, 11, [9], momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, c02_b12_models, 256, reinforce, projection, only_thresholded))
		list_of_conf.append(Configuration(lr, 11, [9], momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, c4_models, 512, reinforce, projection, only_thresholded))
		list_of_conf.append(Configuration(lr, 15, [9], momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, c02_b12_models, 256, reinforce, projection, only_thresholded))
		list_of_conf.append(Configuration(lr, 15, [9], momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, c4_models, 512, reinforce, projection, only_thresholded))
		list_of_conf.append(Configuration(lr, 20, [9], momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, c02_b12_models, 256, reinforce, projection, only_thresholded))
		list_of_conf.append(Configuration(lr, 20, [9], momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, c4_models, 512, reinforce, projection, only_thresholded))
		
	elif sys.argv[1] == "prova":
		#dataset = CIFAR100
		#lr = [1e-3, 2e-4]
		dataset = CIFAR10(batch_size)
		lr = [3e-3, 5e-4]
		branch_size = 256
		threshold = 0.0
		only_thresholded = False
		reduction = 'mean'
		
		#list_of_models = [HCNN3_c0_b2, HCNN3_c1_b2, HCNN3_c2_b2, HCNN3_c3_b2, HCNN3_c4_b2]
		list_of_models = [HCNN3_c0_b2]
		
		list_of_conf.append(Configuration(lr, 20, [9], momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, list_of_models, branch_size, reinforce, projection, only_thresholded))
		
	else:
		raise ValueError(f'Test for {sys.argv[1]} is not supported yet.')


	
	t = Tester(list_of_conf)
	
	t.launch()
	
	t.write_legend("models/" + str(dataset) + "/configurations_legend.txt")
