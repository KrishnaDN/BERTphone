#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:15:31 2020

@author: krishna
"""

import os
import numpy as np
import json
import torch

def pad_labels(labels,blank_symbol=0):
    max_len=100
    input_len=len(labels)
    if input_len<max_len:    
        pad_len=max_len-input_len
        for k in range(pad_len):
            labels.append(0)
    return labels
    

def feature_stack(feature):
    feat_dim,time=feature.shape
    stacked_feats=[]
    for i in range(0,time-3,3):
        splice = feature[:,i:i+3]
        stacked_feats.append(np.array(splice).flatten())
    return np.asarray(stacked_feats)


def SpecAugment(stacked_feature):
    time,feat_dim=stacked_feature.shape
    ##### Masking 5% of the data
    win_len = round(time*0.05)
    mask_start_index = np.random.randint(0, time-win_len)
    create_zero_mat = np.zeros((win_len,feat_dim))
    stacked_feature[mask_start_index:mask_start_index+win_len,:] = create_zero_mat
    masked_features = stacked_feature
    return masked_features

def load_data(json_filepath):
    with open(json_filepath) as f:
        data = json.load(f)
    feature_path = data['feature_path']
    features = np.load(feature_path) 
    stacked_feats = feature_stack(features)
    original_features = stacked_feats.copy()
    masked_features = SpecAugment(stacked_feats)
    phn_seq = [int(item) for item in data['phn_seq'].split(' ')]
    phn_seq_len = data['phn_seq_len']
    final_phn_seq = pad_labels(phn_seq)
    return masked_features,original_features,final_phn_seq,phn_seq_len



def pad_sequence_feats(features_list):
    lengths =[feature_mat.shape[0]  for feature_mat in features_list]
    max_length = max(lengths)
    padded_feat_batch=[]
    for feature_mat in features_list:
        pad_mat = torch.zeros((max_length-feature_mat.shape[0],feature_mat.shape[1]))
        padded_feature = torch.cat((feature_mat,pad_mat),0)
        padded_feat_batch.append(padded_feature)
    return padded_feat_batch




def speech_collate(batch):
    masked_feats = []
    gt_feats = []
    targets_lengths=[]
    targets = []
    for sample in batch:
        masked_feats.append(sample['masked_feats'])
        gt_feats.append(sample['gt_feats'])
        targets.append((sample['phn_seq']))
        targets_lengths.append(sample['labels_length'])
    
    masked_feats_padded = pad_sequence_feats(masked_feats)
    gt_feats_padded = pad_sequence_feats(gt_feats)
    
    return masked_feats_padded,gt_feats_padded, targets,targets_lengths







