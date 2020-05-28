#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 23:28:56 2020

@author: krishna
"""

import os
import glob
import argparse
import json
from utils import load_data
import numpy as np




class FeatureExtraction(object):
    def __init__(self,config):
        super(FeatureExtraction, self).__init__()
        self.dataset_path = config.dataset_path
        self.feature_store_path = config.feature_store_path
        self.feature_type = config.feature
        self.feature_dim = config.feature_dim
    
    def extract_features(self,filepath):
        features = load_data(filepath, win_length=400, sr=16000,hop_length=160, feature_type=self.feature_type,feature_dim=self.feature_dim)
        return features
    
    def process_train_data(self):
        json_files = sorted(glob.glob(os.path.join(self.dataset_path,'TRAIN')+'/*.json'))
        store_folder = os.path.join(self.feature_store_path,'TRAIN')
        if not os.path.exists(store_folder):
            os.makedirs(store_folder)
        for json_filepath in json_files:
            with open(json_filepath) as f:
                data = json.load(f)
            audio_filepath = data['audio_filepath']
            features = self.extract_features(audio_filepath)
            print('Extracting features for {}'.format(audio_filepath))
            store_filepath = os.path.join(store_folder,audio_filepath.split('/')[-1][:-4]+'.npy')
            np.save(store_filepath,features)
            data['feature_path'] = store_filepath
            with open(json_filepath, 'w') as fid:
                json.dump(data, fid,indent=4)
    
    def process_test_data(self):
        json_files = sorted(glob.glob(os.path.join(self.dataset_path,'TEST')+'/*.json'))
        store_folder = os.path.join(self.feature_store_path,'TEST')
        if not os.path.exists(store_folder):
            os.makedirs(store_folder)
        for json_filepath in json_files:
            with open(json_filepath) as f:
                data = json.load(f)
            audio_filepath = data['audio_filepath']
            print('Extracting features for {}'.format(audio_filepath))
            features = self.extract_features(audio_filepath)
            store_filepath = os.path.join(store_folder,audio_filepath.split('/')[-1][:-4]+'.npy')
            np.save(store_filepath,features)
            data['feature_path'] = store_filepath
            with open(json_filepath, 'w') as fid:
                json.dump(data, fid,indent=4)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Configuration for data preparation")
    parser.add_argument("--dataset_path", default="/media/newhd/TIMIT/processed_data", type=str,help='Dataset path')
    parser.add_argument("--feature_store_path", default="/media/newhd/TIMIT/Features", type=str,help='Save directory after processing')
    parser.add_argument("--feature", default="mfcc", type=str,help='Feature type')
    parser.add_argument("--feature_dim", default=13, type=int,help='feature dimensions')

    config = parser.parse_args()

    feat_class = FeatureExtraction(config)
    feat_class.process_train_data()
    feat_class.process_test_data()
        