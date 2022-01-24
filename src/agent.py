import torch
import models
import utils
import math
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
import torch.nn as nn
from collections import OrderedDict
import copy



class Agent():
    def __init__(self, id, args, train_dataset, data_idxs, criterion):
        self.id = id
        self.args = args
        self.criterion = criterion
        
    
        self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)
        # for backdoor attack, agent poisons his local dataset
        self.n_data = len(self.train_dataset)
        if args.attack == 2 and self.id >= args.num_agents - self.args.num_corrupt:
            utils.poison_dataset(train_dataset, args, data_idxs)
        print(f'{self.id} -> Total_Data:{self.n_data}, Dist:{self.train_dataset.class_count}')
        
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True,\
            num_workers=args.num_workers, pin_memory=True)
        self.model = models.get_model(args.data).to(args.device)
        self.opt = torch.optim.Adam(self.model.parameters())
        
        
    def local_train(self, global_model_params):
        """ Do a local training over the received global model, return the update """
        vector_to_parameters(copy.deepcopy(global_model_params), self.model.parameters())
        self.model.train()
       
        for _ in range(self.args.local_ep):
            for _, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(device=self.args.device, non_blocking=True),\
                                 labels.to(device=self.args.device, non_blocking=True)
                                               
                outputs = self.model(inputs)
                minibatch_loss = self.criterion(outputs, labels)
                
                self.opt.zero_grad()
                minibatch_loss.backward()
                #nn.utils.clip_grad_norm_(self.model.parameters(), 10) # to prevent exploding gradients
                self.opt.step()
                                            
        with torch.no_grad():
            return parameters_to_vector(self.model.parameters()) - global_model_params
            
