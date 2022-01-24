#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(1, '../src/')
import torch
from tqdm.notebook import tqdm
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import utils
from torchsummary import summary
from options import args_parser
import models
import torch.nn.functional as F

from time import ctime
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


# In[2]:


args = args_parser()


# getting datasets
tr_ds, te_ds = utils.get_datasets(args.data)
tr_ds, v_ds = torch.utils.data.random_split(tr_ds, [len(tr_ds)-5000, 5000])


tr_loader = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)
v_loader = DataLoader(v_ds, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)
te_loader = DataLoader(te_ds, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=False)


# model etc.
model = models.get_model(args.data).to(args.device)
criterion = nn.CrossEntropyLoss().to(args.device)
opt = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, mode='min', verbose=True)
es = utils.EarlyStopping(patience=10, verbose=True)



start_time, end_time = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
file_name = f"""time:{ctime()}_data:{args.data}"""
writer = SummaryWriter('logs/' + file_name)


start_time.record()
for rnd in tqdm(range(1, args.rounds+1)):
    model.train()
    tr_loss, tr_acc = 0.0, 0.0 
    for _, (inputs, labels) in enumerate(tr_loader):
        inputs, labels = inputs.to(device=args.device, non_blocking=True),labels.to(device=args.device, non_blocking=True)
        
        outputs = model(inputs)
        minibatch_loss = criterion(outputs, labels)
        opt.zero_grad()
        minibatch_loss.backward()
        opt.step()

        # keep track of round loss/accuracy
        tr_loss += minibatch_loss.item()*outputs.shape[0]
        _, pred_labels = torch.max(outputs, 1)
        tr_acc += torch.sum(torch.eq(pred_labels.view(-1), labels)).item()

    # inference after epoch 
    with torch.no_grad():
        tr_loss, tr_acc = tr_loss/len(tr_loader.dataset), tr_acc/len(tr_loader.dataset)       
        v_loss, (v_acc, v_per_class_acc) = utils.get_loss_n_accuracy(model, criterion, v_loader, args)
        v_best_class_acc, v_worst_class_acc = torch.max(v_per_class_acc), torch.min(v_per_class_acc)
        # log/print data
        writer.add_scalar('Training/Loss', tr_loss, rnd)
        writer.add_scalar('Training/Accuracy', tr_acc, rnd)
        writer.add_scalar('Val/Loss', v_loss, rnd)
        writer.add_scalar('Val/Accuracy', v_acc, rnd)
        writer.add_scalar('Val/Best_Class_Acc', v_best_class_acc, rnd)
        writer.add_scalar('Val/Worst_Class_Acc', v_worst_class_acc, rnd)
        writer.add_scalar('Val/Fairness', v_best_class_acc - v_worst_class_acc, rnd)
        # print(f'|Train/Test Loss: {tr_loss:.3f} / {v_loss:.3f}|', end='--')
        # print(f'|Train/Test Acc: {tr_acc:.3f} / {v_acc:.3f}|', end='\r')
        
        scheduler.step(v_loss)
        es(v_loss)
        if es.early_stop:
            print("Early stopping")
            break


# In[8]:


#fin
_, (te_acc, te_per_class_acc) = utils.get_loss_n_accuracy(model, criterion, te_loader, args)
te_best_class_acc, te_worst_class_acc = torch.max(v_per_class_acc), torch.min(v_per_class_acc)
writer.add_scalar('Test/Accuracy', te_acc, 1)
writer.add_scalar('Test/Best_Class_Acc', te_best_class_acc, 1)
writer.add_scalar('Test/Worst_Class_Acc', te_worst_class_acc, 1)
writer.add_scalar('Test/Fairness', te_best_class_acc - te_worst_class_acc, 1)
writer.flush()
writer.close()

end_time.record()
torch.cuda.synchronize()
time_elapsed_secs = start_time.elapsed_time(end_time)/10**3
time_elapsed_mins = time_elapsed_secs/60
print(f'Training took {time_elapsed_secs:.2f} seconds / {time_elapsed_mins:.2f} minutes')

