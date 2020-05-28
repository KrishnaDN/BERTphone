#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 18:54:36 2020

@author: krishna
"""

import os
import glob
import argparse
from data_parser import phoneme_ids,read_pron_dict
import json



class TIMIT(object):
    def __init__(self,config):
        super(TIMIT, self).__init__()
        self.timit_root = config.timit_dataset_root
        self.store_path = config.timit_save_root
        self.phn_list = config.cmu_symbols
        self.cmu_dict = config.cmu_dict
        
    
    def read_dictionary(self):
        self.phn_mapping = phoneme_ids(self.phn_list)    
        self.dictionary = read_pron_dict(self.cmu_dict,self.phn_mapping)
        return self.dictionary,self.phn_mapping
        
    def create_phn_seq(self,word_file,dictionary,phn_mapping):
        read_words = [line.rstrip('\n') for line in open(word_file)]
        phn_seq=[]
        
        for item in read_words:
            word =item.split(' ')[-1]
            try:
                phns = self.dictionary[word]
            except:
                continue
            (phns)
            phn_seq = phn_seq+phns+[self.phn_mapping[' ']]
        return phn_seq
            
    def process_data_train(self):
        dictionary,phn_mapping = self.read_dictionary()
        self.train_dir = os.path.join(self.timit_root,'TRAIN')
        train_subfolders = sorted(glob.glob(self.train_dir+'/*/'))
        for sub_folder in train_subfolders:
            speaker_folders = sorted(glob.glob(sub_folder+'/*/'))
            for spk_folder in speaker_folders:
                store_folder = os.path.join(self.store_path,'TRAIN')
                if not os.path.exists(store_folder):
                    os.makedirs(store_folder)
                WAV_files = sorted(glob.glob(spk_folder+'/*.WAV'))
                for audio_filepath in WAV_files:
                    wrd_file = audio_filepath[:-4]+'.WRD'
                    phn_seq = self.create_phn_seq(wrd_file,dictionary,phn_mapping)
                    json_write_filepath =store_folder+'/'+sub_folder.split('/')[-2]+'_'+spk_folder.split('/')[-2]+'_'+audio_filepath.split('/')[-1][:-4]+'.json'
                    data_frame = {}
                    data_frame['audio_filepath'] = audio_filepath
                    data_frame['phn_seq'] = ' '.join([str(phn_item) for phn_item in phn_seq])
                    data_frame['phn_seq_len']=len(phn_seq)
                    with open(json_write_filepath, 'w') as fid:
                        json.dump(data_frame, fid,indent=4)
        
    def process_data_test(self):
        dictionary,phn_mapping = self.read_dictionary()
        self.test_dir = os.path.join(self.timit_root,'TEST')
        test_subfolders = sorted(glob.glob(self.test_dir+'/*/'))
        for sub_folder in test_subfolders:
            speaker_folders = sorted(glob.glob(sub_folder+'/*/'))
            for spk_folder in speaker_folders:
                store_folder = os.path.join(self.store_path,'TEST')
                if not os.path.exists(store_folder):
                    os.makedirs(store_folder)
                WAV_files = sorted(glob.glob(spk_folder+'/*.WAV'))
                for audio_filepath in WAV_files:
                    wrd_file = audio_filepath[:-4]+'.WRD'
                    phn_seq = self.create_phn_seq(wrd_file,dictionary,phn_mapping)
                    json_write_filepath =store_folder+'/'+sub_folder.split('/')[-2]+'_'+spk_folder.split('/')[-2]+'_'+audio_filepath.split('/')[-1][:-4]+'.json'
                    data_frame = {}
                    data_frame['audio_filepath'] = audio_filepath
                    data_frame['phn_seq'] = ' '.join([str(phn_item) for phn_item in phn_seq])
                    data_frame['phn_seq_len']=len(phn_seq)
                    with open(json_write_filepath, 'w') as fid:
                        json.dump(data_frame, fid,indent=4)
        
        
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Configuration for data preparation")
    parser.add_argument("--timit_dataset_root", default="/media/newhd/TIMIT/data/lisa/data/timit/raw/TIMIT", type=str,help='Dataset path')
    parser.add_argument("--timit_save_root", default="/media/newhd/TIMIT/processed_data", type=str,help='Save directory after processing')
    parser.add_argument("--cmu_dict", default="/home/krishna/Krishna/BERTphone/cmudict.dict", type=str,help='CMU pronounciation directory path')
    parser.add_argument("--cmu_symbols", default="/home/krishna/Krishna/BERTphone/cmudict.symbols", type=str,help='Phoneme list')

    config = parser.parse_args()

    timit = TIMIT(config)
    timit.process_data_train()
    timit.process_data_test()
    

