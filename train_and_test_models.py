from telegramBot import Terminator
import HCNN3
from HCNN3 import HCNN3_c0_b0_r
from cnn3 import CIFAR100, CIFAR10
from cnn3 import device
from tqdm import tqdm 


HCNN3.cnn3.torch.autograd.set_detect_anomaly(False);
HCNN3.cnn3.torch.autograd.profiler.emit_nvtx(False);
HCNN3.cnn3.torch.autograd.profiler.profile(False);

learning_rate = [1e-3, 2e-4]
momentum = 0.9
nesterov = True
epochs = 15
every_print = 64
switch_point = [9]
batch_size = 128
custom_training = False
threshold = 0.6
reduction = 'none'
err = False
track = True


class Configuration:
	
	def __init__(self, batch_size, learning_rate, momentum, nesterov, epochs, every_print, switch_point, custom_training, threshold, reduction, track, dataset, model)
		
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.nesterov = nesterov
		self.epochs = epochs
		self.every_print = every_print
		self.custom_training = custom_training
		self.threshold = threshold
		self.reduction = reduction
		self.track = track
		self.dataset = dataset
		self.batch_size = batch_size
		self.model = model


class Tester():

	def __init__(self, list_of_conf):
	
		self.bot = Terminator()
		self.list_of_conf = list_of_conf
		
	
	def launch(self):
	
		for conf in tqdm(self.list_of_conf, desc = "Testing configuration"):
			cnn = conf.model(conf.learning_rate, conf.momentum, conf.nesterov, conf.dataset(conf.batch_size), 
					conf.epochs, conf.every_print, conf.switch_point, custom_training = conf.custom_training, threshold = conf.threshold, reduction = conf.reduction)
					
			cnn.to(device)
			
			filename = "models/" + str(cnn.dataset) + "/" + str(cnn)
			
			try:
    			cnn.train_model(conf.track, conf.filename)
				cnn.save_model(filename)
				msg = cnn.test(mode = "write", filename = filename)
				cnn.write_configuration(filename, "reinforce\n\n" + msg)
				
			except Exception as error:
				self.bot.sendMessage("Programma NON terminato correttamente\nTipo di errore: " + error.__class__.__name__ + "\nMessaggio: " + str(error))
    			raise error


		self.bot.sendMessage("Test completato correttamente")


try:
    cnn.train_model(track, filename)
    cnn.save_model(filename)
    msg = cnn.test(mode = "write", filename = filename)
    cnn.write_configuration(filename, "reinforce\n\n" + msg)
    
except Exception as errore:
    err = errore

if err is False:
    bot.sendMessage("Programma terminato correttamente\n\n\nPerformance:\n\n" + msg)
else:
    bot.sendMessage("Programma NON terminato correttamente\nTipo di errore: " + err.__class__.__name__ + "\nMessaggio: " + str(err))
    raise err
