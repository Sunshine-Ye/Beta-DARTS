import collections
from collections import defaultdict

import numpy as np
import torch

Node = collections.namedtuple('Node', ['id', 'name'])


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        out = torch.Tensor(inputs.cuda(), **kwargs)
    else:
        out = torch.Tensor(inputs, **kwargs)
    return out
