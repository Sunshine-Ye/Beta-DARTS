from optimizers.darts.genotypes import PRIMITIVES
from optimizers.darts.model_search import Network, Cell, ChoiceBlock, MixedOp
from optimizers.darts.operations import *


def channel_shuffle(x, groups):
    """
    https://github.com/yuhuixu1993/PC-DARTS/blob/86446d1b6bbbd5f752cc60396be13d2d5737a081/model_search.py#L9
    """

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
    """
    Adapted from PCDARTS:
    https://github.com/yuhuixu1993/PC-DARTS/blob/86446d1b6bbbd5f752cc60396be13d2d5737a081/model_search.py#L25
    """

    def __init__(self, C, stride):
        super(MixedOpPCDARTS, self).__init__(C, stride)
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C // 4, stride, False)
            '''
            Not used in NASBench
            if 'pool' in primitive:
              op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            '''
            self._ops.append(op)

    def forward(self, x, weights):
        # channel proportion k=4
        dim_2 = x.shape[1]
        xtemp = x[:, :  dim_2 // 4, :, :]
        xtemp2 = x[:, dim_2 // 4:, :, :]
        temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))
        ans = torch.cat([temp1, xtemp2], dim=1)
        ans = channel_shuffle(ans, 4)
        return ans


class ChoiceBlockPCDARTS(ChoiceBlock):
    def __init__(self, C_in):
        super(ChoiceBlockPCDARTS, self).__init__(C_in)
        # Use the PC_DARTS Mixed Op instead of the DARTS Mixed op
        self.mixed_op = MixedOpPCDARTS(C_in, stride=1)


class CellPCDARTS(Cell):
    def __init__(self, steps,C_prev, C, layer, search_space):
        super(CellPCDARTS, self).__init__(steps, C_prev, C, layer, search_space)
        # Create the choice block.
        self._choice_blocks = nn.ModuleList()
        for i in range(self._steps):
            # Use the PC_DARTS cell instead of the DARTS cell
            choice_block = ChoiceBlockPCDARTS(C_in=C)
            self._choice_blocks.append(choice_block)


class PCDARTSNetwork(Network):
    def __init__(self, C, num_classes, layers, criterion, output_weights, search_space, steps=4):
        super(PCDARTSNetwork, self).__init__(C, num_classes, layers, criterion, output_weights, search_space,
                                             steps=steps)

        # Override the cells module list of DARTS with GDAS variants
        self.cells = nn.ModuleList()
        C_curr = C
        C_prev = C_curr
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                # Double the number of channels after each down-sampling step
                # Down-sample in forward method
                C_curr *= 2

            cell = CellPCDARTS(steps, C_prev, C_curr, layer=i, search_space=search_space)
            self.cells += [cell]
            C_prev = C_curr

    def new(self):
        model_new = PCDARTSNetwork(self._C, self._num_classes, self._layers, self._criterion,
                                   steps=self.search_space.num_intermediate_nodes, output_weights=self._output_weights,
                                   search_space=self.search_space).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new
