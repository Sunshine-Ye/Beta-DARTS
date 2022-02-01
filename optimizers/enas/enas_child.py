import logging

import torch
from torch import nn

from optimizers.darts import utils
from optimizers.enas.data import get_loaders
from optimizers.random_search_with_weight_sharing.darts_wrapper_discrete import DartsWrapper


class ENASChild(DartsWrapper):
    def __init__(self, controller, *args, **kwargs):
        super(ENASChild, self).__init__(*args, **kwargs)
        self.args['entropy_weight'] = 0.0001
        self.args['adam_learning_rate'] = 0.00035
        self.args['bl_dec'] = 0.99
        self.args['weight_decay'] = 1e-4

        self.controller = controller

        self.train_queue, self.reward_queue, self.valid_queue = get_loaders(self.args)

        self.baseline = 0
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            self.args.learning_rate,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, float(self.args.epochs), eta_min=self.args.learning_rate_min)

        self.controller_optimizer = torch.optim.Adam(
            controller.parameters(),
            self.args.adam_learning_rate,
            betas=(0.1, 0.999),
            eps=1e-3,
        )

    def train_model(self, epoch):
        self.objs = utils.AvgrageMeter()
        self.top1 = utils.AvgrageMeter()
        self.top5 = utils.AvgrageMeter()
        for step, (input, target) in enumerate(self.train_queue):
            self.model.train()

            input = input.cuda()
            target = target.cuda()

            self.optimizer.zero_grad()
            self.controller.eval()

            # Sample an architecture from the controller
            arch, _, _ = self.controller()
            arch_parameters = self.get_weights_from_arch(arch)
            self.set_arch_model_weights(arch_parameters)

            # Evaluate the architecture
            logits = self.model(input, discrete=True)
            loss = self.criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()

            n = input.size(0)
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            self.objs.update(loss.data.item(), n)
            self.top1.update(prec1.data.item(), n)
            self.top5.update(prec5.data.item(), n)

            if step % self.args.report_freq == 0:
                logging.info('train %03d %e %f %f', step, self.objs.avg, self.top1.avg, self.top5.avg)
            self.scheduler.step()

        valid_err = self.evaluate(arch)
        logging.info('epoch %d  |  train_acc %f  |  valid_acc %f' % (epoch, self.top1.avg, 1 - valid_err))
        return self.top1.avg

    def train_controller(self):
        total_loss = utils.AvgrageMeter()
        total_reward = utils.AvgrageMeter()
        total_entropy = utils.AvgrageMeter()

        for step in range(300):
            input, target = self.reward_queue.next_batch()
            self.model.eval()
            n = input.size(0)

            input = input.cuda()
            target = target.cuda()

            self.controller_optimizer.zero_grad()

            self.controller.train()
            # Sample an architecture from the controller and plug it into the one-shot model.
            arch, log_prob, entropy = self.controller()
            arch_parameters = self.get_weights_from_arch(arch)
            self.set_arch_model_weights(arch_parameters)

            with torch.no_grad():
                # Make sure that no gradients are propagated through the one-shot model
                # for the controller updates
                logits = self.model(input, discrete=True).detach()
                reward = utils.accuracy(logits, target)[0]

            if self.args.entropy_weight is not None:
                reward += self.args.entropy_weight * entropy

            log_prob = torch.sum(log_prob)
            if self.baseline is None:
                self.baseline = reward
            self.baseline = self.args.bl_dec * self.baseline + (1 - self.args.bl_dec) * reward

            loss = log_prob * (reward - self.baseline)
            loss = loss.mean()

            loss.backward()

            self.controller_optimizer.step()

            total_loss.update(loss.item(), n)
            total_reward.update(reward.item(), n)
            total_entropy.update(entropy.item(), n)

            if step % self.args.report_freq == 0:
                logging.info('controller %03d %e %f %f', step, total_loss.avg, total_reward.avg, self.baseline.item())

    def evaluate_sampled_architecture(self):
        arch, log_prob, entropy = self.controller()
        return self.evaluate(arch)

    def evaluate(self, arch, split=None):
        # Return error since we want to minimize obj val
        # logging.info(arch)
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        weights = self.get_weights_from_arch(arch)
        self.set_arch_model_weights(weights)

        self.model.eval()
        self.controller.eval()

        if split is None:
            n_batches = 1
        else:
            n_batches = len(self.valid_queue)

        for step in range(n_batches):
            input, target = self.valid_queue.next_batch()
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = self.model(input, discrete=True)
            loss = self.criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            # if step % self.args.report_freq == 0:
                # logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        return 1 - 0.01 * top1.avg
