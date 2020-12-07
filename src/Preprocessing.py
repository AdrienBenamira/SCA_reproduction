import torch
import numpy as np

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        trace, sensitive = sample['trace'], sample['sensitive']

        return {'trace': torch.from_numpy(trace),
                'sensitive': torch.from_numpy(np.array([sensitive]))}


class Horizontal_Scaling_0_1(object):

    def __call__(self, sample):
        trace, sensitive = sample['trace'], sample['sensitive']

        scale = 1.0 / (torch.max(trace).item() - torch.min(trace).item())
        trace = trace.sub(torch.min(trace).item()).mul(scale)

        return {'trace': trace,
                'sensitive': sensitive}

class Horizontal_Scaling_m1_1(object):

    def __call__(self, sample):
        trace, sensitive = sample['trace'], sample['sensitive']

        scale = 1.0 / (torch.max(trace).item() - torch.min(trace).item())
        trace = trace.sub(torch.min(trace).item()).mul(scale)
        trace = trace.mul(1- (-1)).add(-1)
        return {'trace': trace,
                'sensitive': sensitive}


