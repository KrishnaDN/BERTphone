#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 19:47:07 2020

@author: krishna
"""

import os


def phoneme_ids(phonemes_list):
    read_phone_list = [line.rstrip('\n') for line in open(phonemes_list)]
    phoneme_dict ={}
    id_=0
    phoneme_dict[' '] = id_
    for item in read_phone_list:
        id_+=1
        phoneme_dict[item] = id_
    return phoneme_dict


def read_pron_dict(pron_dict_path,phoneme_dict):
    read_data = [line.rstrip('\n') for line in open(pron_dict_path)]
    cmu_dict = {}
    for item in read_data:
        word = item.split(' ')[0]
        phn_ids = []
        for phn in item.split(' ')[1:]:
            try:
                phn_ids.append(phoneme_dict[phn])
            except:
                continue
        cmu_dict[word] = phn_ids
    return cmu_dict