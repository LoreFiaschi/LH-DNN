import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
from torch.linalg import vector_norm as vnorm
from torch.linalg import solve as solve_matrix_system

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Lambda

from abc import ABC
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 


num_cores = 8
torch.set_num_interop_threads(num_cores) # Inter-op parallelism
torch.set_num_threads(num_cores) # Intra-op parallelism

num_class_c1 = 2
num_class_c2 = 7
num_class_c3 = 10

#--- coarse 1 classes ---
labels_c_1 = ('transport', 'animal')
#--- coarse 2 classes ---
labels_c_2 = ('sky', 'water', 'road', 'bird', 'reptile', 'pet', 'medium')
#--- fine classes ---
labels_c_3 = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def c3_to_c1(y):
    if y < 2 or y > 7:
        return 0
    return 1

def c3_to_c2(y):
    match y:
        case 0:
            return 0
        case 1:
            return 2
        case 2:
            return 3
        case 3:
            return 5
        case 4:
            return 6
        case 5:
            return 5
        case 6:
            return 4
        case 7:
            return 6
        case 8:
            return 1
        case _:
            return 2

def c2_to_c1(y):
    if y < 3:
        return 0
    return 1



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

coarser = Lambda(lambda y: torch.tensor([c3_to_c1(y), c3_to_c2(y), int(y)]))

batch_size = 128

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=False, transform=transform, target_transform = coarser)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_cores)

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform, target_transform = coarser)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_cores)






class CNN3(ABC, nn.Module):
    def __init__(self, learning_rate, momentum, nesterov, trainloader, testloader, 
                 epochs, num_class_c1, num_class_c2, num_class_c3, labels_c_1, labels_c_2, labels_c_3, 
                 every_print = 512, switch_point = None, custom_training = False, training_size = 50000):
        
        super().__init__()
        self.trainloader = trainloader
        self.testloader = testloader
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.activation = F.relu
        self.class_levels = 3
        self.num_c_1 = num_class_c1
        self.num_c_2 = num_class_c2
        self.num_c_3 = num_class_c3
        self.epochs = epochs
        self.switch_point = switch_point if (switch_point is None or switch_point < epochs) else epochs
        self.custom_training = custom_training
        self.labels_c_1 = labels_c_1
        self.labels_c_2 = labels_c_2
        self.labels_c_3 = labels_c_3
        self.every_print = every_print - 1 # assumed power of 2, -1 to make the mask
        self.track_size = int( training_size / batch_size / every_print ) 

    
    def forward(self, x):
        pass


    # Assumption: W is column full rank. 
    def project(self, z): # https://math.stackexchange.com/questions/4021915/projection-orthogonal-to-two-vectors

        W1 = self.layerb17.weight.clone().detach()
        W2 = self.layerb27.weight.clone().detach()
        ort2 = torch.empty_like(z)
        ort3 = torch.empty_like(z)

        for i, zi in enumerate(z):
            Rk = torch.diag(torch.where(zi.clone().detach() != 0, 1.0, 0.0))
            W1k = W1.mm(Rk)
            W2k_ = W2.mm(Rk)
            W2k = torch.vstack((W1k, W2k_))
            ort2[i,:] = self.compute_othogonal(zi, W1k)
            ort3[i,:] = self.compute_othogonal(zi, W2k)
            
        #prj2 = z.clone().detach() - ort2.clone().detach()
        #prj3 = z.clone().detach() - ort3.clone().detach()
        
        return ort2, ort3, #prj2, ort2, prj3, ort3

    def compute_othogonal(self, z, W, eps = 1e-8):
        WWT = torch.matmul(W, W.T)
        P = solve_matrix_system(WWT + torch.randn_like(WWT) * eps, torch.eye(W.size(0)))
        P = torch.matmul(P, W)
        P = torch.eye(W.size(1)) - torch.matmul(W.T, P)
        
        return torch.matmul(z, P)
        
    
    def predict_and_learn(self, batch, labels):
        self.optimizer.zero_grad()
        predict = self(batch)
        loss_f = self.criterion(predict[0], labels[:,0])
        loss_i1 = self.criterion(predict[1], labels[:,1])
        loss_i2 = self.criterion(predict[2], labels[:,2])
        
        loss_f.backward(retain_graph=True)
        loss_i1.backward(retain_graph=True)
        loss_i2.backward()

        self.optimizer.step()

        return torch.tensor([loss_f, loss_i1, loss_i2])


    def train_model(self, verbose = False):
        self.train()
        
        for epoch in tqdm(np.arange(self.epochs), desc="Training: "):
            #self.update_training_params(epoch)

            if verbose:
                running_loss = torch.zeros(self.class_levels)
            
            for iter, (batch, labels) in enumerate(self.trainloader):
                loss = self.predict_and_learn(batch, labels)

                if verbose:
                    running_loss += (loss - running_loss) / (iter+1)
                    if (iter + 1) & self.every_print == 0:
                        print(f'[{epoch + 1}] loss_f : {running_loss[0] :.3f}')
                        print(f'[{epoch + 1}] loss_i1: {running_loss_[1] :.3f}')
                        print(f'[{epoch + 1}] loss_i2: {running_loss_[2] :.3f}')
                        for i in np.arange(self.class_levels):
                            running_loss[i] = 0.0


    def training_loop_body(self):
        running_loss = torch.zeros(self.class_levels)
            
        for iter, (batch, labels) in enumerate(self.trainloader):
            loss = self.predict_and_learn(batch, labels)

            running_loss += (loss - running_loss) / (iter+1)
            if (iter + 1) & self.every_print == 0:
                self.loss_track[self.num_push, :] = running_loss
                self.accuracy_track[self.num_push, :] = self.test(mode = "train")
                self.num_push += 1
                for i in np.arange(self.class_levels):
                        running_loss[i] = 0.0

    
    def train_track(self, filename = ""):
        self.train()
        
        self.loss_track = torch.zeros(self.epochs * self.track_size, self.class_levels)
        self.accuracy_track = torch.zeros(self.epochs * self.track_size, self.class_levels)
        self.num_push = 0

        if self.custom_training:
          self.custom_training_f()
            
        elif self.switch_point is None:
            for epoch in tqdm(np.arange(9), desc="Training: "):
                self.training_loop_body()
                
        else:
            for epoch in tqdm(np.arange(self.switch_point), desc="Training: "):
                self.training_loop_body()
    
            self.optimizer.param_groups[0]['lr'] = self.learning_rate[1]
    
            for epoch in tqdm(np.arange(self.switch_point, self.epochs), desc="Training: "):
                self.training_loop_body()

        self.plot_training_loss(filename+"_train_loss.pdf")
        self.plot_test_accuracy(filename+"_test_accuracy_.pdf")

        
        def custom_training_f(self):
            pass


    def initialize_memory(self):
        self.correct_c1_pred = torch.zeros(self.num_c_1)
        self.total_c1_pred = torch.zeros_like(self.correct_c1_pred)
        
        self.correct_c2_pred = torch.zeros(self.num_c_2)
        self.total_c2_pred = torch.zeros_like(self.correct_c2_pred)
        
        self.correct_c3_pred = torch.zeros(self.num_c_3)
        self.total_c3_pred = torch.zeros_like(self.correct_c3_pred)

        self.correct_c1_vs_c2_pred = torch.zeros(self.num_c_1)
        self.total_c1_vs_c2_pred = torch.zeros_like(self.correct_c1_vs_c2_pred)

        self.correct_c2_vs_c3_pred = torch.zeros(self.num_c_2)
        self.total_c2_vs_c3_pred = torch.zeros_like(self.correct_c2_vs_c3_pred)

        self.correct_c1_vs_c3_pred = torch.zeros(self.num_c_1)
        self.total_c1_vs_c3_pred = torch.zeros_like(self.correct_c1_vs_c3_pred)

    
    def collect_test_performance(self):
        with torch.no_grad():
            for images, labels in self.testloader:
                predictions = self(images)
                predicted = torch.zeros(predictions[0].size(0), self.class_levels, dtype=torch.long)
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

                    if predicted[i,1] == c3_to_c2(predicted[i,2]):
                        self.correct_c2_vs_c3_pred[predicted[i,1]] += 1

                    if predicted[i,0] == c3_to_c1(predicted[i,2]):
                        self.correct_c1_vs_c3_pred[predicted[i,0]] += 1

                    if predicted[i,0] == c2_to_c1(predicted[i,1]):
                        self.correct_c1_vs_c2_pred[predicted[i,0]] += 1
                        
                    self.total_c1_pred[labels[i,0]] += 1
                    self.total_c2_pred[labels[i,1]] += 1
                    self.total_c3_pred[labels[i,2]] += 1
                    self.total_c1_vs_c3_pred[predicted[i,0]] += 1
                    self.total_c1_vs_c2_pred[predicted[i,0]] += 1
                    self.total_c2_vs_c3_pred[predicted[i,1]] += 1


    def test_results_to_text(self, ):
        str = ""
        
        # accuracy for each class        
        for i in np.arange(self.num_c_1):
            accuracy_c1 = 100 * float(self.correct_c1_pred[i]) / self.total_c1_pred[i]
            str += f'Accuracy for class {self.labels_c_1[i]:5s}: {accuracy_c1:.2f} %'
            str += '\n'

        str += '\n'
        
        for i in np.arange(self.num_c_2):
            accuracy_c2 = 100 * float(self.correct_c2_pred[i]) / self.total_c2_pred[i]
            str += f'Accuracy for class {self.labels_c_2[i]:5s}: {accuracy_c2:.2f} %'
            str += '\n'
            
        str += '\n'
        
        for i in np.arange(self.num_c_3):
            accuracy_c3 = 100 * float(self.correct_c3_pred[i]) / self.total_c3_pred[i]
            str += f'Accuracy for class {self.labels_c_3[i]:5s}: {accuracy_c3:.2f} %'
            str += '\n'
            
        # accuracy for the whole dataset
        str += '\n'

        str += f'Accuracy on c1: {(100 * self.correct_c1_pred.sum() / self.total_c1_pred.sum()):.2f} %'
        str += '\n'

        str += f'Accuracy on c2: {(100 * self.correct_c2_pred.sum() / self.total_c2_pred.sum()):.2f} %'
        str += '\n'

        str += f'Accuracy on c3: {(100 * self.correct_c3_pred.sum() / self.total_c3_pred.sum()):.2f} %'
        str += '\n'
        
        str += '\n'

        # cross classes accuracy (tree)
        for i in np.arange(self.num_c_1):
            accuracy_c1_c2 = 100 * float(self.correct_c1_vs_c2_pred[i]) / self.total_c1_vs_c2_pred[i]
            str += f'Cross-accuracy {self.labels_c_1[i]:9s} vs c2: {accuracy_c1_c2:.2f} %'
            str += '\n'
            
        str += '\n'
        
        for i in np.arange(self.num_c_2):
            accuracy_c2_c3 = 100 * float(self.correct_c2_vs_c3_pred[i]) / self.total_c2_vs_c3_pred[i]
            str += f'Cross-accuracy {self.labels_c_2[i]:7s} vs c3: {accuracy_c2_c3:.2f} %'
            str += '\n'
            
        str += '\n'
        
        for i in np.arange(self.num_c_1):
            accuracy_c1_c3 = 100 * float(self.correct_c1_vs_c3_pred[i]) / self.total_c1_vs_c3_pred[i]
            str += f'Cross-accuracy {self.labels_c_1[i]:9s} vs c3: {accuracy_c1_c3:.2f} %'
            str += '\n'

        return str


    def barplot(self, x, accuracy, labels, title):
        plt.bar(x, accuracy, tick_label = labels)
        plt.xlabel("Classes")
        plt.ylabel("Accuracy")
        plt.title(title)
        plt.show();

    
    def plot_test_results(self):
        # accuracy for each class
        accuracy_c1 = torch.empty(self.num_c_1)
        for i in np.arange(self.num_c_1):
            accuracy_c1[i] = float(self.correct_c1_pred[i]) / self.total_c1_pred[i]
        self.barplot(np.arange(self.num_c_1), accuracy_c1, self.labels_c_1, "Accuracy on the first level")

        accuracy_c2 = torch.empty(self.num_c_2 + 1)
        for i in np.arange(self.num_c_2):
            accuracy_c2[i] = float(self.correct_c2_pred[i]) / self.total_c2_pred[i]
        accuracy_c2[self.num_c_2] = self.correct_c2_pred.sum() / self.total_c2_pred.sum()
        self.barplot(np.arange(self.num_c_2 + 1), accuracy_c2, (*self.labels_c_2, 'overall'), "Accuracy on the second level")

        accuracy_c3 = torch.empty(self.num_c_3 + 1)
        for i in np.arange(self.num_c_3):
            accuracy_c3[i] = float(self.correct_c3_pred[i]) / self.total_c3_pred[i]
        accuracy_c3[self.num_c_3] = self.correct_c3_pred.sum() / self.total_c3_pred.sum()
        self.barplot(np.arange(self.num_c_3 + 1), accuracy_c3, (*self.labels_c_3, 'overall'), "Accuracy on the third level")

    
    def test(self, mode = "print", filename = None):
        self.initialize_memory()
        self.eval()

        self.collect_test_performance()

        match mode:
            case "plot":
                self.plot_test_results()

            case "print":
                msg = self.test_results_to_text()
                print(m)
                return msg

            case "write":
                msg = self.test_results_to_text()
                with open(filename+"_test_performance.txt", 'w') as f:
                    f.write(msg)
                return msg

            case "train":
                accuracy_c1 = self.correct_c1_pred.sum() / self.total_c1_pred.sum()
                accuracy_c2 = self.correct_c2_pred.sum() / self.total_c2_pred.sum()
                accuracy_c3 = self.correct_c3_pred.sum() / self.total_c3_pred.sum()

                self.train()

                return torch.tensor([accuracy_c1, accuracy_c2, accuracy_c3])
                
            case _:
                raise AttributeError("Test mode not available")
        
    
    def plot_training_loss(self, filename = None):
        plt.figure(figsize=(12, 6))
        plt.plot(np.linspace(1, self.epochs, self.loss_track.size(0)), self.loss_track[:, 0].numpy(), label = "First level")
        plt.plot(np.linspace(1, self.epochs, self.loss_track.size(0)), self.loss_track[:, 1].numpy(), label = "Second level")
        plt.plot(np.linspace(1, self.epochs, self.loss_track.size(0)), self.loss_track[:, 2].numpy(), label = "Third level")
        plt.title("Training loss")
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.xticks(np.linspace(1, self.epochs, self.epochs)[0::2])
        plt.legend()
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.show();

    
    def plot_test_accuracy(self, filename = None):
        plt.figure(figsize=(12, 6))
        plt.plot(np.linspace(1, self.epochs, self.accuracy_track.size(0)), self.accuracy_track[:, 0].numpy(), label = "First level")
        plt.plot(np.linspace(1, self.epochs, self.accuracy_track.size(0)), self.accuracy_track[:, 1].numpy(), label = "Second level")
        plt.plot(np.linspace(1, self.epochs, self.accuracy_track.size(0)), self.accuracy_track[:, 2].numpy(), label = "Third level")
        plt.title("Test accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.xticks(np.linspace(1, self.epochs, self.epochs)[0::2])
        plt.legend()
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.show();

    
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
        msg += "Switch point: " + str(self.switch_point) + "\n\n"
        msg += "Number of params: " + str(self.num_param()) + "\n\n"
        msg += "Additional info: " + additional_info
        
        with open(filename+"_configuration.txt", 'w') as f:
            f.write(msg)
