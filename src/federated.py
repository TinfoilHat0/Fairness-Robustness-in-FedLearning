import torch 
import utils
import models
import math
import copy
import numpy as np
from agent import Agent
from tqdm import tqdm
from options import args_parser
from aggregation import Aggregation
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
import torch.nn as nn
from time import ctime
import random

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    args = args_parser()
    
    # load dataset and user groups (i.e., user to data mapping)
    train_dataset, val_dataset = utils.get_datasets(args.data)
    if not args.concent:
        user_groups = utils.distribute_data(train_dataset, args)
    else:
        user_groups = utils.distribute_data_dirichlet(train_dataset, args)
        
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    
    # poison stuff
    cum_poison_acc_mean = 0
    idxs = [idx for idx in range(len(val_dataset))]
    poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
    utils.poison_dataset(poisoned_val_set.dataset, args, poison_all=True)
    poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)                                        
        
        
    # agents, models etc.    
    global_model = models.get_model(args.data).to(args.device)
    criterion = nn.CrossEntropyLoss().to(args.device)
    es = utils.EarlyStopping(patience=10, verbose=True)
    agents = []
    for _id in range(0, args.num_agents):
        agent = Agent(_id, args, train_dataset, user_groups[_id], criterion)
        agents.append(agent) 
    aggregator = Aggregation(args, n_params=len(parameters_to_vector(global_model.parameters())))
    
    # data recorders
    start_time, end_time = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    file_name = f"""time:{ctime()}_p:{args.poison_frac}_concent:{args.concent}_data:{args.data}_nAgents:{args.num_agents}_aggrFun:{args.aggr}_rlrThresh:{args.rlr_threshold}_numCorrupt:{args.num_corrupt}"""
    writer = SummaryWriter('logs/' + file_name)
    
    # main training loop
    start_time.record()
    for rnd in tqdm(range(1, args.rounds+1)):
        agent_updates = []
        rnd_global_model_params = parameters_to_vector(global_model.parameters()).detach()
        for agent_id in np.random.choice(args.num_agents, math.ceil(args.num_agents*args.agent_frac), replace=False):
            update = agents[agent_id].local_train(rnd_global_model_params)
            agent_updates.append((agent.n_data, update))
            
        # aggregate params, and update the global model at then end of round
        new_global_params = aggregator.aggregate_updates(rnd_global_model_params, agent_updates)
        vector_to_parameters(new_global_params, global_model.parameters())
        
        
        # inference in every args.snap rounds
        if rnd % args.snap == 0:
            with torch.no_grad():
                val_loss, (val_acc, val_per_class_acc) = utils.get_loss_n_accuracy(global_model, criterion, val_loader, args)
                val_best_class_acc, val_worst_class_acc = torch.max(val_per_class_acc), torch.min(val_per_class_acc)
                writer.add_scalar('Val/Loss', val_loss, rnd)
                writer.add_scalar('Val/Accuracy', val_acc, rnd)
                writer.add_scalar('Val/Best_Class_Acc', val_best_class_acc, rnd)
                writer.add_scalar('Val/Worst_Class_Acc', val_worst_class_acc, rnd)
                writer.add_scalar('Val/Fairness', val_best_class_acc - val_worst_class_acc, rnd)
                
                
                poison_loss, (poison_acc, _) = utils.get_loss_n_accuracy(global_model, criterion, poisoned_val_loader, args)
                cum_poison_acc_mean += poison_acc
                #writer.add_scalar('Poison/Base_Class_Accuracy', val_per_class_acc[args.base_class], rnd)
                #writer.add_scalar('Poison/Cumulative_Poison_Accuracy_Mean', cum_poison_acc_mean/rnd, rnd)
                writer.add_scalar('Poison/Poison_Accuracy', poison_acc, rnd)
                writer.add_scalar('Poison/Poison_Loss', poison_loss, rnd)
                    
                
                es(val_loss)
                if es.early_stop:
                    print("Early stopping")
                    break
            
               
    
    # end of the world
    writer.flush()
    writer.close()
    end_time.record()
    torch.cuda.synchronize()
    time_elapsed_secs = start_time.elapsed_time(end_time)/10**3
    time_elapsed_mins = time_elapsed_secs/60
    writer.add_scalar('Time', time_elapsed_mins, 0)
    torch.cuda.empty_cache()
    print(f'Training took {time_elapsed_secs:.2f} seconds / {time_elapsed_mins:.2f} minutes')
    
    
   

    
    
    
      
              