from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings

from src.utils import load_ascad

warnings.filterwarnings("ignore")


class AscadDataLoader_train(Dataset):


    def __init__(self, config, transform=None):

        self.config = config
        self.X_profiling, self.Y_profiling, _, _, self.real_key = load_ascad(config.general.ascad_database_file)
        self.transform = transform

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

class AscadDataLoader_test(Dataset):

    def __init__(self, config, transform=None):
        self.config = config
        _, _, self.X_attack, self.target, self.real_key = load_ascad(config.general.ascad_database_file)
        self.transform = transform

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