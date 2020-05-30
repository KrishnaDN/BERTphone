#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:42:16 2020

@author: krishna
"""

import torch
import numpy as np
from torch.utils.data import DataLoader   
from SpeechDataGenerator import SpeechDataGenerator
import torch.nn as nn
import os
import numpy as np
from torch import optim
import argparse
from models.BERTphone import BERTphone
from sklearn.metrics import accuracy_score
from utils.utility import speech_collate
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')


########## Argument parser
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-training_filepath',type=str,default='meta/training.txt')
parser.add_argument('-testing_filepath',type=str, default='meta/testing.txt')
parser.add_argument('-input_feat_dim', action="store_true", default=39)
parser.add_argument('-num_phones', action="store_true", default=86)
parser.add_argument('-num_heads', action="store_true", default=13)
parser.add_argument('-num_layers', action="store_true", default=12)
parser.add_argument('-lamda_val', action="store_true", default=0.1)
parser.add_argument('-batch_size', action="store_true", default=32)
parser.add_argument('-use_gpu', action="store_true", default=True)
parser.add_argument('-num_epochs', action="store_true", default=100)
args = parser.parse_args()

### Data related
dataset_train = SpeechDataGenerator(manifest=args.training_filepath)
dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,shuffle=True,collate_fn=speech_collate) 

dataset_test = SpeechDataGenerator(manifest=args.testing_filepath)
dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,shuffle=True,collate_fn=speech_collate) 

## Model related
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = BERTphone(args.input_feat_dim,args.num_phones, args.num_heads, args.num_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)
ctc_loss_function = nn.CTCLoss(blank=85,zero_infinity=True, reduction='mean')
rec_loss_function = nn.L1Loss(reduction='mean')



def train(dataloader_train,epoch):
    train_loss_list=[]
    model.train()
    for i_batch, sample_batched in enumerate(dataloader_train):
        masked_features = torch.from_numpy(np.asarray([torch_tensor.numpy().T for torch_tensor in sample_batched[0]])).float()
        original_features = torch.from_numpy(np.asarray([torch_tensor.numpy().T for torch_tensor in sample_batched[1]])).float()
        targets = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[2]])).float()
        target_length = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[3]]))
        input_length = torch.full(size=(targets.shape[0],), fill_value=masked_features.shape[2],dtype=torch.int64)
    
        masked_features, original_features, = masked_features.to(device), original_features.to(device)
        targets = targets.to(device)
        target_length,input_length = target_length.to(device),input_length.to(device)
        masked_features.requires_grad = True
        optimizer.zero_grad()
        
        ctc_logits,projection_rec,transformer_out = model(masked_features)
        #### CTC loss
        ctc_loss = ctc_loss_function(ctc_logits, targets, input_length, target_length)
        #### Reconstruction loss
        
        rec_loss = rec_loss_function(projection_rec,original_features )
    
        ##### Total loss
        total_loss = args.lamda_val * np.sqrt(masked_features.shape[2])*rec_loss + (1-args.lamda_val)*ctc_loss 
        total_loss.backward()
        
        optimizer.step()
        
        train_loss_list.append(total_loss.item())
        #train_acc_list.append(accuracy)
        #if i_batch%100==0:
        #    print('Loss {} after {} iteration'.format(np.mean(np.asarray(train_loss_list)),i_batch))
        
        
    mean_loss = np.mean(np.asarray(train_loss_list))
    print('Total training loss {} after {} epochs'.format(mean_loss,epoch))
    
    
    
def test(dataloader_test,epoch):
    model.eval()
    with torch.no_grad():
        test_loss_list=[]
        for i_batch, sample_batched in enumerate(dataloader_test):
            masked_features = torch.from_numpy(np.asarray([torch_tensor.numpy().T for torch_tensor in sample_batched[0]])).float()
            original_features = torch.from_numpy(np.asarray([torch_tensor.numpy().T for torch_tensor in sample_batched[1]])).float()
            targets = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[2]])).float()
            target_length = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[3]]))
            input_length = torch.full(size=(targets.shape[0],), fill_value=masked_features.shape[2],dtype=torch.int64)
        
            
            masked_features, original_features, = masked_features.to(device), original_features.to(device)
            targets = targets.to(device)
            target_length,input_length = target_length.to(device),input_length.to(device)
            ctc_logits,projection_rec,transformer_out = model(masked_features)
            #### CTC loss
            ctc_loss = ctc_loss_function(ctc_logits, targets, input_length, target_length)
            #### Reconstruction loss
            rec_loss = rec_loss_function(projection_rec,original_features )
            ##### Total loss
            total_loss = args.lamda_val * np.sqrt(masked_features.shape[2])*rec_loss + (1-args.lamda_val)*ctc_loss 
            test_loss_list.append(total_loss.item())
        
        
        mean_loss = np.mean(np.asarray(test_loss_list))
        print('Total testing loss {} after {} epochs'.format(mean_loss,epoch))
        model_save_path = os.path.join('save_model', 'best_check_point_'+str(epoch)+'_'+str(mean_loss))
        state_dict = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}
        torch.save(state_dict, model_save_path)
    
if __name__ == '__main__':
    for epoch in range(args.num_epochs):
        train(dataloader_train,epoch)
        test(dataloader_test,epoch)
        


    
