from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn import preprocessing, model_selection

# Ignore warnings
import warnings

from src.utils import load_ascad

warnings.filterwarnings("ignore")


class AscadDataLoader_train(Dataset):
    def __init__(self, config, transform=None, feature_scaler = None):
        self.config = config
        self.X_profiling, self.Y_profiling, _, _, _ = load_ascad(config.general.ascad_database_file)
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

    def train_validation_split(self, test_size = 0.1):
        self.X_profiling, X_profiling_validation, self.Y_profiling, Y_profiling_validation = model_selection.train_test_split(self.X_profiling, self.Y_profiling, test_size = test_size, random_state =0)
        return [X_profiling_validation, Y_profiling_validation]

    def to_categorical(self, num_classes):
        self.Y_profiling = np.eye(num_classes, dtype='uint8')[self.Y_profiling]

    def feature_min_max_scaling(self, a, b):
        scaler = preprocessing.MinMaxScaler(feature_range=(a, b))
        self.X_profiling = scaler.fit_transform(self.X_profiling)
        self.feature_scaler = scaler

    def feature_standardization(self):
        scaler = preprocessing.StandardScaler()
        self.X_profiling = scaler.fit_transform(self.X_profiling)
        self.X_profiling_validation = scaler.fit(self.X_profiling_validation)
        self.feature_scaler = scaler

    def get_feature_scaler(self):
        return self.feature_scaler


class AscadDataLoader_validation(Dataset):

    def __init__(self, config, X_profiling_validation, Y_profiling_validation, transform=None, feature_scaler = None):
        self.config = config
        self.X_profiling, self.Y_profiling = X_profiling_validation, Y_profiling_validation
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

    def feature_scaling(self, feature_scaler = None):
        if self.feature_scaler == None and feature_scaler == None:
            return "No feature scaler"
        elif feature_scaler != None:
            self.X_profiling = feature_scaler.transform(self.X_profiling)
        else:
            self.X_profiling = self.feature_scaler.transform(self.X_profiling)




class AscadDataLoader_test(Dataset):

    def __init__(self, config, transform=None, feature_scaler = None):
        self.config = config
        _, _, self.X_attack, self.targets, self.real_key = load_ascad(config.general.ascad_database_file)
        self.transform = transform
        self.feature_scaler = feature_scaler

    def __len__(self):
        return len(self.X_attack)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        trace = self.X_attack[idx]
        target = self.targets[idx] ##All possible S(key xor x) for all the 256 keys. 256
        sample = {'trace': trace, 'sensitive': target}

        if self.transform:
            sample = self.transform(sample)

        return sample


    def to_categorical(self, num_classes):
        self.targets = np.eye(num_classes, dtype='uint8')[self.targets]


    def feature_scaling(self, feature_scaler = None):
        if self.feature_scaler == None and feature_scaler == None:
            return "No feature scaler"
        elif feature_scaler != None:
            self.X_attack = feature_scaler.transform(self.X_attack)
        else:
            self.X_attack = self.feature_scaler.transform(self.X_attack)

    def rank(self, ModelPredictions, ntraces = 300, interval = 10):
        ranktime = np.zeros(int(ntraces / interval))
        pred = np.zeros(256)

        idx = np.random.randint(ModelPredictions.shape[0], size=ntraces)

        for i, p in enumerate(idx):
            for k in range(ModelPredictions.shape[1]): #256
                pred[k] += ModelPredictions[p][self.targets[p, k]]


            if i % interval == 0:
                ranked = np.argsort(pred)[::-1]
                ranktime[int(i / interval)] = list(ranked).index(self.real_key)
        return ranktime
