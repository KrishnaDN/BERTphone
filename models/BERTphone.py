#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:37:04 2020

@author: krishna
"""



import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from modules.transformer import TransformerEncoder

class BERTphone(nn.Module):
    def __init__(self,input_feat_dim, num_phones=86,out_feat_dim=13,num_heads = 13, layers = 12):
        super(BERTphone, self).__init__()
        self.input_feat_dim = 39
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = 0.1
        self.relu_dropout = 0.1
        self.res_dropout = 0.0
        self.out_dropout = 0.0
        self.embed_dropout = 0.1
        self.attn_mask = False
        self.num_phones=num_phones
        self.out_feature_dim =out_feat_dim
        
        #self.lstm = nn.LSTM(256,256,dropout=0.2,batch_first=True)
        self.transformer = self.get_transformer()        
        # self.drop = nn.Dropout(p=0.2)
        self.proj_layer_ctc = nn.Conv1d(39,self.num_phones, kernel_size=1, padding=0, bias=False)
        self.proj_layer_rec = nn.Conv1d(39,self.out_feature_dim, kernel_size=1, padding=0, bias=False)
        
        
    def get_transformer(self, layers=-1):
        
        embed_dim, attn_dropout = self.input_feat_dim, self.attn_dropout
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
    
    
    
    def forward(self, x0):
    
        x = x0.permute(2,0,1)
        transformer_out = self.transformer(x)
        transformer_out = transformer_out.permute(1,2,0)
        projection_ctc = F.log_softmax(self.proj_layer_ctc(transformer_out),dim=1)
        projection_rec = self.proj_layer_rec(transformer_out)
        ctc_logits = projection_ctc.permute(2,0,1)
        return ctc_logits,projection_rec
        


