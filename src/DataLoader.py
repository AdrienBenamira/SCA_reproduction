from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn import preprocessing

# Ignore warnings
import warnings

from src.utils import load_ascad

warnings.filterwarnings("ignore")


class AscadDataLoader_train(Dataset):


    def __init__(self, config, transform=None, feature_scaler = None):

        self.config = config
        self.X_profiling, self.Y_profiling, _, _, self.real_key = load_ascad(config.general.ascad_database_file)
        self.transform = transform
        self.feature_scaler = feature_scaler

    def __len__(self):
        return len(self.X_profiling)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        trace = self.X_profiling[idx]
        sensitive = self.Y_profiling[idx]
        sample = {'trace': trace, 'sensitive': sensitive}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def feature_min_max_scaling(self, a, b):
        scaler = preprocessing.MinMaxScaler(feature_range=(a, b))
        self.X_profiling = scaler.fit_transform(self.X_profiling)
        self.feature_scaler = scaler

    def feature_standardization(self):
        scaler = preprocessing.StandardScaler()
        self.X_profiling = scaler.fit_transform(self.X_profiling)
        self.feature_scaler = scaler


    def get_feature_scaler(self):
        return self.feature_scaler

class AscadDataLoader_test(Dataset):

    def __init__(self, config, transform=None, feature_scaler = None):
        self.config = config
        _, _, self.X_attack, self.target, self.real_key = load_ascad(config.general.ascad_database_file)
        self.transform = transform
        self.feature_scaler = feature_scaler

    def __len__(self):
        return len(self.X_attack)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        trace = self.X_attack[idx]
        sensitive = self.target[idx]
        sample = {'trace': trace, 'sensitive': sensitive}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def feature_scaling(self, feature_scaler = None):
        if self.feature_scaler == None and feature_scaler == None:
            return "No feature scaler"
        elif feature_scaler != None:
            self.X_attack = feature_scaler.transform(self.X_attack)
        else:
            self.X_attack = self.feature_scaler.transform(self.X_attack)