#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 12:16:28 2020

@author: krishna
"""


import numpy as np
import torch
from utils import utility

class SpeechDataGenerator():
    """Speech dataset."""

    def __init__(self, manifest):
        """
        Read the textfile and get the paths
        """
        self.json_links = [line.rstrip('\n').split(' ')[0] for line in open(manifest)]
        
    def __len__(self):
        return len(self.json_links)
    
    
    def __getitem__(self, idx):
        json_link =self.json_links[idx]
        masked_features,original_feats,final_phn_seq,phn_seq_len = utility.load_data(json_link)
        #lang_label=lang_id[self.audio_links[idx].split('/')[-2]]
        sample = {'masked_feats': torch.from_numpy(np.ascontiguousarray(masked_features)), 
                  'gt_feats': torch.from_numpy(np.ascontiguousarray(original_feats)),
                  'phn_seq': torch.from_numpy(np.ascontiguousarray(final_phn_seq)),
                  'labels_length': torch.from_numpy(np.ascontiguousarray(phn_seq_len))}
        return sample
    
    
