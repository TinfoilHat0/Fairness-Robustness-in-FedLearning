import torch
import copy
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from math import floor, sqrt, pi
from collections import defaultdict, Counter
from torch.distributions.normal import Normal
import random
import cv2
from torch.nn.utils import vector_to_parameters, parameters_to_vector

class DatasetSplit(Dataset):
    """ An abstract Dataset class wrapped around Pytorch Dataset class """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs
        self.targets = torch.Tensor([self.dataset.targets[idx] for idx in idxs])
        self.per_class_count()
        
    def classes(self):
        return torch.unique(self.targets)   
    
    def per_class_count(self, n_class=10):
        self.class_count = {}
        # assuming 10 classes in dataset
        for c in range(n_class):
            c_count = len(self.targets[self.targets == c])
            self.class_count[c] = c_count
        
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        inp, target = self.dataset[self.idxs[item]]
        return inp, target
    
    

def distribute_data_dirichlet(dataset, args, n_class=10):
    num_clean_agents = args.num_agents - args.num_corrupt
    
    # partition[c][i] is the fraction of samples agent i gets from class i
    partition = np.random.dirichlet([args.concent]*num_clean_agents, size=n_class)
    
    # sort labels
    labels_sorted = dataset.targets.sort()
    # create a list of pairs (index, label), i.e., at index we have an instance of  label
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
    labels_dict = defaultdict(list)
    for k, v in class_by_labels:
        labels_dict[k].append(v)

    dict_users = defaultdict(list)
    if args.num_corrupt == 1:
        num_clean_agents = args.num_agents - 1
        adv_idxs = random.sample(range(len(dataset)), int(len(dataset)*0.1))
        dict_users[args.num_agents-1] = adv_idxs
        for c in range(10):
            labels_dict[c] = list(set(labels_dict[c]) - (set(labels_dict[c]) & set(adv_idxs)))
        
    for c in range(n_class):
        # num of samples of class c in dataset
        n_classC_items = len(labels_dict[c]) 
        for i in range(num_clean_agents):
            # num. of samples agent i gets from class c
            n_agentI_items = int(partition[c][i] * n_classC_items)
            dict_users[i] += labels_dict[c][:n_agentI_items]
            del labels_dict[c][:n_agentI_items]
        # if any class c item remains due to flooring, give em to first agent
        dict_users[0] += labels_dict[c]
        
    return dict_users        




def distribute_data(dataset, args, n_classes=10):
    if args.num_agents == 1:
        return {0:range(len(dataset))}
    
    # sort labels
    labels_sorted = dataset.targets.sort()
    # create a list of pairs (index, label), i.e., at index we have an instance of  label
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
    labels_dict = defaultdict(list)
    for k, v in class_by_labels:
        labels_dict[k].append(v)
    
    dict_users = defaultdict(list)
    if args.num_corrupt == 1:
        num_clean_agents = args.num_agents - 1
        adv_idxs = random.sample(range(len(dataset)), int(len(dataset)*0.1))
        dict_users[args.num_agents-1] = adv_idxs
        for c in range(10):
            labels_dict[c] = list(set(labels_dict[c]) - (set(labels_dict[c]) & set(adv_idxs)))
    
    # split indexes to shards
    shard_size = len(dataset) // (num_clean_agents * args.class_per_agent)
    slice_size = (len(dataset) // n_classes) // shard_size    
    for k, v in labels_dict.items():
        labels_dict[k] = chunker_list(v, slice_size)
    
    # distribute shards to users
    for user_idx in range(num_clean_agents):
        class_ctr = 0
        for j in range(0, n_classes):
            if class_ctr == args.class_per_agent:
                    break
            elif len(labels_dict[j]) > 0:
                dict_users[user_idx] += labels_dict[j][0]
                del labels_dict[j%n_classes][0]
                class_ctr+=1

    return dict_users       



def get_datasets(data, augment=True):
    """ returns train and test datasets """
    train_dataset, test_dataset = None, None
    data_dir = '../data'
    
    if data == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    elif data == 'fmnist':
        transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)   
    
    elif data == 'cifar10':
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(test_dataset.targets)  
    return train_dataset, test_dataset    



def chunker_list(seq, size):
    return [seq[i::size] for i in range(size)]



def poison_dataset(dataset, args, data_idxs=None, poison_all=False):
    all_idxs = [idx for idx in range(len(dataset))]
    if data_idxs != None:
       all_idxs = list(set(all_idxs).intersection(data_idxs))
        
    poison_frac = 1 if poison_all else args.poison_frac    
    poison_idxs = random.sample(all_idxs, floor(poison_frac*len(all_idxs)))
    for idx in poison_idxs:
        clean_img = dataset.data[idx]
        bd_img = add_pattern_bd(clean_img, args)
        dataset.data[idx] = torch.tensor(bd_img)
        dataset.targets[idx] = args.target_class    
    return



def add_pattern_bd(x, args, size=5, pixel_value=255):
    """
    Augments a matrix by setting a checkboard-like pattern of values some `distance` away from the bottom-right
    edge to 1. Works for single images or a batch of images.
    :param x: N X W X H matrix or W X H matrix. will apply to last 2
    :type x: `np.ndarray`
    :param distance: distance from bottom-right walls. defaults to 2
    :type distance: `int`
    :param pixel_value: Value used to replace the entries of the image matrix
    :type pixel_value: `int`
    :return: augmented matrix
    :rtype: np.ndarray
    """
    x = np.array(x)
    
    if args.pattern_type == 'plus':
        start_idx = 5
        # vertical line  
        for i in range(start_idx, start_idx+size):
            x[i, start_idx] = pixel_value 
        
        # horizontal line
        for i in range(start_idx-size//2, start_idx+size//2 + 1):
            x[start_idx+size//2, i] = pixel_value
    
    elif args.pattern_type == 'apple':
        trojan = None
        if args.data == 'mnist':
            trojan = cv2.imread('../src/apple.png', cv2.IMREAD_GRAYSCALE)
            trojan = cv2.bitwise_not(trojan)
            trojan = cv2.resize(trojan, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
        
        elif args.data == 'cifar10':
            trojan = cv2.imread('../src/apple.png')
            trojan = cv2.bitwise_not(trojan)
            trojan = cv2.resize(trojan, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
        
        x = cv2.addWeighted(x, 1, trojan, 1,0)
          
    return x


def get_loss_n_accuracy(model, criterion, data_loader, args, adv=True, num_classes=10):
    """ Returns the loss and total accuracy, per class accuracy on the supplied data loader """
    
    # disable BN stats during inference
    model.eval()                                      
    total_loss, correctly_labeled_samples = 0, 0
    confusion_matrix = torch.zeros(num_classes, num_classes)
            
    # forward-pass to get loss and predictions of the current batch
    for _, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device=args.device, non_blocking=True),\
                labels.to(device=args.device, non_blocking=True)
                
        
        if not adv:
            labels.fill_(args.base_class)
                                            
        # compute the total loss over minibatch
        outputs = model(inputs)
        avg_minibatch_loss = criterion(outputs, labels)
        total_loss += avg_minibatch_loss.item()*outputs.shape[0]
                        
        # get num of correctly predicted inputs in the current batch
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, labels)).item()
        # fill confusion_matrix
        for t, p in zip(labels.view(-1), pred_labels.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
                                
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correctly_labeled_samples / len(data_loader.dataset)
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    return avg_loss, (accuracy, per_class_accuracy)



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    """ Taken from https://github.com/Bjarten/early-stopping-pytorch """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
   
    def __call__(self, val_loss, model=None):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model=None):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        #torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
