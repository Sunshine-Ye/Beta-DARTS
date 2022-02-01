import torch
from torch.autograd import Variable
from torch import autograd
import sys
sys.path.insert(0, '../../')
from optimizers.darts.architect import Architect


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class ArchitectGDAS(Architect):

    def __init__(self, model, args):
        self.grad_clip = args.grad_clip
        super(ArchitectGDAS, self).__init__(model, args)

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            self._backward_step(input_valid, target_valid)

        # Add gradient clipping for gdas because gumbel softmax leads to gradients with high magnitude
        torch.nn.utils.clip_grad_norm_(self.model.arch_parameters(), self.grad_clip)
        self.optimizer.step()

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self.model._loss(input, target)
        theta = _concat(self.model.parameters()).data
        # Changes to reflect that for unused ops there will be no gradient and this needs to be handled
        dtheta = _concat(
            [grad_i + self.network_weight_decay * theta_i if grad_i is not None else self.network_weight_decay * theta_i
             for grad_i, theta_i in
             zip(torch.autograd.grad(loss, self.model.parameters(), allow_unused=True), self.model.parameters())])

        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).musl_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(dtheta)
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))
        return unrolled_model

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        # Changes to reflect that for unused ops there will be no gradient and this needs to be handled
        vector = [v.grad.data if v.grad is not None else torch.zeros_like(v) for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)
