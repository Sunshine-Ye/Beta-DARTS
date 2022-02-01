import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from optimizers.darts.genotypes import PRIMITIVES

"""
Implementation done by MengTianjian reimplementation by https://github.com/MengTianjian/enas-pytorch/blob/master/micro_controller.py
Modified by ANONYMOUS
"""


class Controller(torch.nn.Module):
    def __init__(self, search_space, args):
        torch.nn.Module.__init__(self)
        self.search_space = search_space
        self.args = args
        # TODO
        self.child_num_ops = args.child_num_ops
        self.num_cells = args.child_num_cells
        self.lstm_size = args.lstm_size
        self.lstm_num_layers = args.lstm_num_layers
        self.lstm_keep_prob = args.lstm_keep_prob
        self.temperature = args.temperature
        self.tanh_constant = args.controller_tanh_constant
        self.op_tanh_reduce = args.controller_op_tanh_reduce

        self.encoder = nn.Embedding(self.child_num_ops + 1, self.lstm_size)

        self.lstm = nn.LSTMCell(self.lstm_size, self.lstm_size)
        self.w_soft = nn.Linear(self.lstm_size, self.child_num_ops, bias=False)
        b_soft = torch.zeros(1, self.child_num_ops)
        b_soft[:, 0:2] = 10
        self.b_soft = nn.Parameter(b_soft)
        b_soft_no_learn = np.array([0.25, 0.25] + [-0.25] * (self.child_num_ops - 2))
        b_soft_no_learn = np.reshape(b_soft_no_learn, [1, self.child_num_ops])
        self.b_soft_no_learn = torch.Tensor(b_soft_no_learn).requires_grad_(False).cuda()
        # attention
        self.w_attn_1 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.w_attn_2 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)
        self.reset_param()

    def reset_param(self):
        for name, param in self.named_parameters():
            if 'b_soft' not in name:
                nn.init.uniform_(param, -0.1, 0.1)

    def forward(self):
        arc_seq_1, entropy_1, log_prob_1, c, h = self.run_sampler(use_bias=True)
        sample_arc = arc_seq_1
        sample_entropy = entropy_1
        sample_log_prob = log_prob_1
        return sample_arc, sample_log_prob, sample_entropy

    def run_sampler(self, prev_c=None, prev_h=None, use_bias=False):
        if prev_c is None:
            # TODO: multi-layer LSTM
            prev_c = torch.zeros(1, self.lstm_size).cuda()
            prev_h = torch.zeros(1, self.lstm_size).cuda()

        inputs = self.encoder(torch.zeros(1).long().cuda())

        anchors = []
        anchors_w_1 = []

        for layer_id in range(2):
            embed = inputs
            next_h, next_c = self.lstm(embed, (prev_h, prev_c))
            prev_c, prev_h = next_c, next_h
            anchors.append(torch.zeros(next_h.shape).cuda())
            anchors_w_1.append(self.w_attn_1(next_h))

        layer_id = 2
        entropy = []
        log_prob = []

        num_nodes = len(self.search_space.num_parents_per_node)
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        adjacency_matrix[0, 1] = 1
        node_list = []

        while layer_id < len(self.search_space.num_parents_per_node):
            prev_layers = []

            # # 1. Sample the parents for each choice block
            embed = inputs
            next_h, next_c = self.lstm(embed, (prev_h, prev_c))
            prev_c, prev_h = next_c, next_h
            query = torch.stack(anchors_w_1[:layer_id], dim=1)
            query = query.view(layer_id, self.lstm_size)
            query = torch.tanh(query + self.w_attn_2(next_h))
            query = self.v_attn(query)
            logits = query.view(1, layer_id)
            if self.temperature is not None:
                logits /= self.temperature
            if self.tanh_constant is not None:
                logits = self.tanh_constant * torch.tanh(logits)
            prob = F.softmax(logits, dim=-1)

            # Sample the required number of parents from the multinomial distribution
            indices = prob.multinomial(num_samples=self.search_space.num_parents_per_node[str(layer_id)],
                                       replacement=False)
            for parent in indices.cpu().numpy()[0]:
                adjacency_matrix[parent, layer_id] = 1

            curr_log_prob = []
            for i in range(indices.size()[1]):
                curr_log_prob.append(F.cross_entropy(logits, indices[0, i].view(1)))

            log_prob.append(sum(curr_log_prob) / len(curr_log_prob))

            curr_ent = -torch.mean(torch.sum(torch.mul(F.log_softmax(logits, dim=-1), prob), dim=1)).detach()

            entropy.append(curr_ent)

            for i in range(indices.size()[1]):
                prev_layers.append(anchors[indices[0, i].view(1)])
            inputs = prev_layers[-1].view(1, -1).requires_grad_()

            # # 2. Sample the operation to be applied to the parents of the choice block
            embed = inputs
            next_h, next_c = self.lstm(embed, (prev_h, prev_c))
            prev_c, prev_h = next_c, next_h
            logits = self.w_soft(next_h) + self.b_soft.requires_grad_()
            if self.temperature is not None:
                logits /= self.temperature
            if self.tanh_constant is not None:
                op_tanh = self.tanh_constant / self.op_tanh_reduce
                logits = op_tanh * torch.tanh(logits)
            if use_bias:
                logits += self.b_soft_no_learn
            prob = F.softmax(logits, dim=-1)
            op_id = prob.multinomial(num_samples=1).long().view(1)
            node_list.append(PRIMITIVES[int(op_id.cpu().numpy())])

            curr_log_prob = F.cross_entropy(logits, op_id)
            log_prob.append(curr_log_prob)
            curr_ent = -torch.mean(torch.sum(torch.mul(F.log_softmax(logits, dim=-1), prob), dim=1)).detach()
            entropy.append(curr_ent)
            inputs = self.encoder(op_id + 1)

            next_h, next_c = self.lstm(inputs, (prev_h, prev_c))
            prev_c, prev_h = next_c, next_h
            anchors.append(next_h)
            anchors_w_1.append(self.w_attn_1(next_h))
            inputs = self.encoder(torch.zeros(1).long().cuda())
            layer_id += 1

        entropy = sum(entropy)
        log_prob = sum(log_prob)
        last_c = next_c
        last_h = next_h

        return (adjacency_matrix, node_list), entropy, log_prob, last_c, last_h
