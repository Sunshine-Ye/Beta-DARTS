import torch.nn.functional as F

from optimizers.darts.model_search import Network, MixedOp, ChoiceBlock, Cell
from optimizers.darts.operations import *


class MixedOpGDAS(MixedOp):
    """
    Adapted from GDAS:
    https://github.com/D-X-Y/GDAS/blob/ea4d245a0eb1d1863418ded661c5867d4669d9bf/lib/nas/model_search_acc2.py
    """

    def __init__(self, *args, **kwargs):
        super(MixedOpGDAS, self).__init__(*args, **kwargs)

    def forward(self, x, weights):
        cpu_weights = weights.tolist()
        use_sum = sum([abs(_) > 1e-10 for _ in cpu_weights])
        if use_sum > 3:
            return sum(w * op(x) for w, op in zip(weights, self._ops))
        else:
            clist = []
            for j, cpu_weight in enumerate(cpu_weights):
                if abs(cpu_weight) > 1e-10:
                    clist.append(weights[j] * self._ops[j](x))
            assert len(clist) > 0, 'invalid length : {:}'.format(cpu_weights)
            return sum(clist)


class ChoiceBlockGDAS(ChoiceBlock):
    """
    Adapted to match Figure 3 in:
    Bender, Gabriel, et al. "Understanding and simplifying one-shot architecture search."
    International Conference on Machine Learning. 2018.
    """

    def __init__(self, C_in):
        super(ChoiceBlockGDAS, self).__init__(C_in)
        # Use the GDAS Mixed Op instead of the DARTS Mixed op
        self.mixed_op = MixedOpGDAS(C_in, stride=1)


class CellGDAS(Cell):

    def __init__(self, steps, C_prev, C, layer, search_space):
        super(CellGDAS, self).__init__(steps, C_prev, C, layer, search_space)
        # Create the choice block.
        self._choice_blocks = nn.ModuleList()
        for i in range(self._steps):
            # Use the GDAS cell instead of the DARTS cell
            choice_block = ChoiceBlockGDAS(C_in=C)
            self._choice_blocks.append(choice_block)


class GDASNetwork(Network):

    def __init__(self, tau, C, num_classes, layers, criterion, output_weights, search_space, steps=4):
        super(GDASNetwork, self).__init__(C, num_classes, layers, criterion, output_weights, 
                                          search_space, steps=steps)
        self.tau = tau

        # Override the cells module list of DARTS with GDAS variants
        self.cells = nn.ModuleList()
        C_curr = C
        C_prev = C_curr
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                # Double the number of channels after each down-sampling step
                # Down-sample in forward method
                C_curr *= 2

            cell = CellGDAS(steps, C_prev, C_curr, layer=i, search_space=search_space)
            self.cells += [cell]
            C_prev = C_curr

    def set_tau(self, tau):
        self.tau = tau

    def new(self):
        model_new = GDASNetwork(self.tau, self._C, self._num_classes, self._layers, self._criterion,
                                steps=self.search_space.num_intermediate_nodes, output_weights=self._output_weights,
                                search_space=self.search_space).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input, discrete=False, normalize=False, updateType=None):
        # NASBench only has one input to each cell
        s0 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if i in [self._layers // 3, 2 * self._layers // 3]:
                # Perform down-sampling by factor 1/2
                # Equivalent to https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_builder.py#L68
                s0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)(s0)
            # If using discrete architecture from random_ws search with weight sharing then pass through architecture
            # weights directly.
            # For GDAS use gumbel softmax hard, therefore per mixed block only a single operation is evaluated
            preprocess_op_mixed_op = lambda x: x if discrete else F.gumbel_softmax(x, tau=self.tau, hard=True, dim=-1)
            # Don't use hard for the rest, because it very quickly gave exploding gradients
            preprocess_op = lambda x: x if discrete else F.gumbel_softmax(x, tau=self.tau, hard=False, dim=-1)

            # Normalize mixed_op weights for the choice blocks in the graph
            mixed_op_weights = preprocess_op_mixed_op(self._arch_parameters[0])
            # Normalize the output weights
            output_weights = preprocess_op(self._arch_parameters[1]) if self._output_weights else None
            # Normalize the input weights for the nodes in the cell
            input_weights = [preprocess_op(alpha) for alpha in self._arch_parameters[2:]]
            s0 = cell(s0, mixed_op_weights, output_weights, input_weights)

        # Include one more preprocessing step here
        s0 = self.postprocess(s0)  # [N, C_max * (steps + 1), w, h] -> [N, C_max, w, h]

        # Global Average Pooling by averaging over last two remaining spatial dimensions
        # Like in nasbench: https://github.com/google-research/nasbench/blob/master/nasbench/lib/model_builder.py#L92
        out = s0.view(*s0.shape[:2], -1).mean(-1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits
