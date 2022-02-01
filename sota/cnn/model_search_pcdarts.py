import torch
import torch.nn as nn
import torch.nn.functional as F
from sota.cnn.operations import *
import sys
sys.path.insert(0, '../../')
from sota.cnn.model_search import MixedOp, Cell, Network


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class MixedOpPCDARTS(MixedOp):

    def __init__(self, C, stride, PRIMITIVES):
        super(MixedOpPCDARTS, self).__init__(C, stride, PRIMITIVES)
        self._ops = nn.ModuleList()
        self.mp = nn.MaxPool2d(2, 2)
        for primitive in PRIMITIVES:
            op = OPS[primitive](C // 4, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C // 4, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
       #channel proportion k=4
        dim_2 = x.shape[1]
        xtemp = x[:, :  dim_2//4, :, :]
        xtemp2 = x[:,  dim_2//4:, :, :]
        temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))
        #reduction cell needs pooling before concat
        if temp1.shape[2] == x.shape[2]:
            ans = torch.cat([temp1, xtemp2], dim=1)
        else:
            ans = torch.cat([temp1, self.mp(xtemp2)], dim=1)
        ans = channel_shuffle(ans, 4)
        #ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
        #except channe shuffle, channel shift also works
        return ans


class CellPCDARTS(Cell):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(CellPCDARTS, self).__init__(steps, multiplier,
                                          C_prev_prev, C_prev, C, reduction, reduction_prev)
        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()

        edge_index = 0

        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOpPCDARTS(C, stride, self.primitives[edge_index])
                self._ops.append(op)
                edge_index += 1


class PCDARTSNetwork(Network):

    def __init__(self, C, num_classes, layers, criterion, primitives, steps=4,
                 multiplier=4, stem_multiplier=3, drop_path_prob=0.0):
        super(PCDARTSNetwork, self).__init__(C, num_classes, layers, criterion, primitives, 
            steps, multiplier, stem_multiplier, drop_path_prob)
        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = CellPCDARTS(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = PCDARTSNetwork(self._C, self._num_classes, self._layers,
                                   self._criterion, self.PRIMITIVES,
                                   drop_path_prob=self.drop_path_prob).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new
    
