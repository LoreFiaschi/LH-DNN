from telegramBot import Terminator
import HVGG
from HVGG import HVGG_c0_b1, HVGG_c0_b2
from HVGG import HVGG_c1_b0_16, HVGG_c1_b1_16, HVGG_c1_b2_16, HVGG_c1_b0_19, HVGG_c1_b1_19, HVGG_c1_b2_19
from HVGG import HVGG_c2_b0_16, HVGG_c2_b1_16, HVGG_c2_b2_16, HVGG_c2_b0_19, HVGG_c2_b1_19, HVGG_c2_b2_19
from HVGG import HVGG_c3_b0_16, HVGG_c3_b1_16, HVGG_c3_b2_16, HVGG_c3_b0_19, HVGG_c3_b1_19, HVGG_c3_b2_19
from fashion_mnist import FashionMnist
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

	HVGG.cnn3.torch.autograd.set_detect_anomaly(False);
	HVGG.cnn3.torch.autograd.profiler.emit_nvtx(False);
	HVGG.cnn3.torch.autograd.profiler.profile(False);
	
	# fixed params
	batch_size = 128
	track = True
	custom_training = False
	momentum = 0.9
	nesterov = True
	every_print = 32
	reinforce = True
	projection = False
	
	
	list_of_conf = []
	
	if sys.argv[1] == "16":
		b0_models = 			[HVGG_c1_b0_16]#, HVGG_c2_b0_16]
		b1_models = [HVGG_c0_b1, HVGG_c1_b1_16]#, HVGG_c2_b1_16]
		b2_models = [HVGG_c0_b2, HVGG_c1_b2_16]#, HVGG_c2_b2_16]
		
	elif sys.argv[1] == "19":
		b0_models = 			[HVGG_c1_b0_19, HVGG_c2_b0_19, HVGG_c3_b0_19]
		b1_models = [HVGG_c0_b1, HVGG_c1_b1_19, HVGG_c2_b1_19, HVGG_c3_b1_19]
		b2_models = [HVGG_c0_b2, HVGG_c1_b2_19, HVGG_c2_b2_16, HVGG_c3_b2_19]
		
	else:
		raise ValueError(f'Test for {sys.argv[1]} is not supported yet.')

	dataset = FashionMnist(batch_size)
	lr2 = [1e-3, 2e-4]
	#lr3 = [1e-3, 2e-4, 5e-5]
	only_thresholded = False
	threshold = 0.0
	reduction = 'mean'
	branch_size = 256

	list_of_conf.append(Configuration(lr2, 7, [4], momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, b0_models+b1_models+b2_models, branch_size, reinforce, projection, only_thresholded))
	#list_of_conf.append(Configuration(lr2, 9, [5], momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, b0_models+b1_models+b2_models, branch_size, reinforce, projection, only_thresholded))
	#list_of_conf.append(Configuration(lr2, 3, [2], momentum, nesterov, every_print, custom_training, threshold, reduction, track, dataset, b1_models+b2_models, branch_size, reinforce, projection, only_thresholded))

	
	t = Tester(list_of_conf)
	
	t.launch()
	
	t.write_legend("models/" + str(dataset) + "/configurations_legend.txt")
