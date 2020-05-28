#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 23:32:55 2020

@author: krishna
"""

import librosa
import numpy as np

def load_wav(audio_filepath, sr):
    audio_data,fs  = librosa.load(audio_filepath,sr=sr)
    return audio_data
        
def mfcc_from_wav(wav, hop_length, win_length,sr, mfcc_dim=13):
    mfcc = librosa.feature.mfcc(wav, sr=sr,hop_length=hop_length, win_length=win_length, n_mfcc=mfcc_dim) 
    return mfcc.T

def filterbank_energy_from_wav(wav, hop_length, win_length,sr, n_mels=40):
    mfcc = librosa.feature.melspectrogram(wav, sr=sr, hop_length=hop_length, win_length=win_length,n_mels=n_mels) 
    return mfcc.T

def spectogram_from_wav(wav, hop_length, win_length, n_fft=512):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T


def load_data(filepath, win_length=400, sr=16000,hop_length=160, feature_type='mfcc',feature_dim=13):
    wav = load_wav(filepath, sr=sr)
    if feature_type=='mfcc':
        features = mfcc_from_wav(wav, hop_length, win_length, sr=sr,mfcc_dim=feature_dim)
    elif feature_type=='mfbe':
        features = filterbank_energy_from_wav(wav, hop_length, win_length, sr=sr,n_mels=feature_dim)
    else:
        features = spectogram_from_wav(wav, hop_length, win_length, n_fft=512)
    
    mag, _ = librosa.magphase(features)  # magnitude
    mag_T = mag.T
    spec_mag = mag_T
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    #return spec_mag
    return (spec_mag - mu) / (std + 1e-5)

