import torch
import numpy as np

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        trace, sensitive = sample['trace'], sample['sensitive']

        return {'trace': torch.from_numpy(trace),
                'sensitive': torch.from_numpy(np.array([sensitive]))}


