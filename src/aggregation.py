import torch
import utils
import models
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import torch.nn as nn
import random
from torch.distributions.normal import Normal
from torch.nn import functional as F
from copy import deepcopy
import math
import copy


class Aggregation():
    def __init__(self,  args, n_params):
        self.args = args
        self.server_lr = args.server_lr
        self.n_params = n_params
        
        
        
    def aggregate_updates(self, global_model_params, agent_updates):
        lr_vector = torch.Tensor([self.server_lr]*self.n_params).to(self.args.device)
        if self.args.rlr_threshold > 0:
            lr_vector = self.compute_robustLR(agent_updates)
        
    
        aggregated_updates = 0
        if self.args.aggr == 'avg':
            aggregated_updates = self.agg_avg(agent_updates)
        elif self.args.aggr=='comed':
            aggregated_updates = self.agg_comed(agent_updates)
        
        new_global_params =  (global_model_params + lr_vector*aggregated_updates).float() 
        return new_global_params          
     
     
     
    def agg_avg(self, agent_updates):
        """ classic fed avg """
        sm_updates, total_data = 0, 0
        for _, (n_data, update) in enumerate(agent_updates):

            sm_updates += n_data * update
            total_data += n_data
        return sm_updates / total_data
    
    
    def agg_comed(self, agent_updates):
        agent_updates_col_vector = [update.view(-1, 1) for _, update in agent_updates]
        concat_col_vectors = torch.cat(agent_updates_col_vector, dim=1)
        return torch.median(concat_col_vectors, dim=1).values
    
    
    def compute_robustLR(self, agent_updates):
        agent_updates_sign = [torch.sign(update) for _, update in agent_updates]  
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        
        sm_of_signs[sm_of_signs < self.args.rlr_threshold] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.args.rlr_threshold] = self.server_lr                                            
        return sm_of_signs.to(self.args.device)
    
    
    
    
    
 