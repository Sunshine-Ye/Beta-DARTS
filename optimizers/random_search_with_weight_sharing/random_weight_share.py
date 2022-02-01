import argparse
import inspect
import json
import logging
import os
import pickle
import shutil
import sys
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn

sys.path.insert(0, '../../')
from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2
from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
from nasbench_analysis import eval_darts_one_shot_model_in_nasbench as naseval
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Args for SHA with weight sharing')
parser.add_argument('--benchmark', dest='benchmark', type=str, default='cnn')
parser.add_argument('--seed', dest='seed', type=int, default=0)
parser.add_argument('--epochs', dest='epochs', type=int, default=100)
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
parser.add_argument('--grad_clip', dest='grad_clip', type=float, default=0.25)
parser.add_argument('--save_dir', dest='save_dir', type=str, default=None)
parser.add_argument('--eval_only', dest='eval_only', type=int, default=0)
# CIFAR-10 only argument.  Use either 16 or 24 for the settings for random_ws search
# with weight-sharing used in our experiments.
parser.add_argument('--init_channels', dest='init_channels', type=int, default=16)
parser.add_argument('--search_space', choices=['1', '2', '3'], default='1')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
args = parser.parse_args()

save_dir = '../../experiments/random_ws/search_{}_{}_{}'.format(
    time.strftime("%Y%m%d-%H%M%S"), args.search_space, args.seed)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if args.eval_only:
    assert args.save_dir is not None

# Dump the config of the run folder
with open(os.path.join(save_dir, 'config.json'), 'w') as fp:
    json.dump(args.__dict__, fp)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter(save_dir + '/runs')


class Rung:
    def __init__(self, rung, nodes):
        self.parents = set()
        self.children = set()
        self.rung = rung
        for node in nodes:
            n = nodes[node]
            if n.rung == self.rung:
                self.parents.add(n.parent)
                self.children.add(n.node_id)


class Node:
    def __init__(self, parent, arch, node_id, rung):
        self.parent = parent
        self.arch = arch
        self.node_id = node_id
        self.rung = rung

    def to_dict(self):
        out = {'parent': self.parent, 'arch': self.arch, 'node_id': self.node_id, 'rung': self.rung}
        if hasattr(self, 'objective_val'):
            out['objective_val'] = self.objective_val
        return out


class Random_NAS:
    def __init__(self, B, model, seed, save_dir):
        self.save_dir = save_dir

        self.B = B
        self.model = model
        self.seed = seed

        self.iters = 0

        self.arms = {}
        self.node_id = 0

    def print_summary(self):
        logging.info(self.parents)
        objective_vals = [(n, self.arms[n].objective_val) for n in self.arms if hasattr(self.arms[n], 'objective_val')]
        objective_vals = sorted(objective_vals, key=lambda x: x[1])
        best_arm = self.arms[objective_vals[0][0]]
        val_ppl = self.model.evaluate(best_arm.arch, split='valid')
        logging.info(objective_vals)
        logging.info('best valid ppl: %.2f' % val_ppl)

    def get_arch(self):
        arch = self.model.sample_arch()
        self.arms[self.node_id] = Node(self.node_id, arch, self.node_id, 0)
        self.node_id += 1
        return arch

    def save(self):
        to_save = {a: self.arms[a].to_dict() for a in self.arms}
        # Only replace file if save successful so don't lose results of last pickle save
        with open(os.path.join(self.save_dir, 'results_tmp.pkl'), 'wb') as f:
            pickle.dump(to_save, f)
        shutil.copyfile(os.path.join(self.save_dir, 'results_tmp.pkl'), os.path.join(self.save_dir, 'results.pkl'))

        self.model.save(epoch=self.model.epochs)

    def run(self):
        epochs = 0
        # self.get_eval_arch(1)
        while self.iters < self.B:
            arch = self.get_arch()
            self.model.train_batch(arch)
            self.iters += 1
            # If epoch has changed then evaluate the network.
            if epochs < self.model.epochs:
                epochs = self.model.epochs
                self.get_eval_arch(1)
            # if self.iters % 500 == 0:
                # self.save()
        # self.save()

    def get_eval_arch(self, epoch, rounds=None):
        # n_rounds = int(self.B / 7 / 1000)
        if rounds is None:
            n_rounds = max(1, int(self.B / 10000))
        else:
            n_rounds = rounds
        best_rounds = []
        for r in range(n_rounds):
            sample_vals = []
            for _ in range(1000):
                arch = self.model.sample_arch()
                try:
                    ppl = self.model.evaluate(arch)
                except Exception as e:
                    ppl = 1000000
                # logging.info(arch)
                # logging.info('objective_val: %.3f' % ppl)
                sample_vals.append((arch, ppl))

            # Save sample validations
            with open(os.path.join(self.save_dir,
                                   'sample_val_architecture_epoch_{}.obj'.format(self.model.epochs)), 'wb') as f:
                pickle.dump(sample_vals, f)

            sample_vals = sorted(sample_vals, key=lambda x: x[1])

            full_vals = []
            if 'split' in inspect.getfullargspec(self.model.evaluate).args:
                for i in range(5):
                    arch = sample_vals[i][0]
                    try:
                        ppl = self.model.evaluate(arch, split='valid')
                    except Exception as e:
                        ppl = 1000000
                    full_vals.append((arch, ppl))
                full_vals = sorted(full_vals, key=lambda x: x[1])
                logging.info('best arch: %s, best arch valid performance: %.3f' % (
                    ' '.join([str(i) for i in full_vals[0][0]]), full_vals[0][1]))
                best_rounds.append(full_vals[0])

                # benchmark
                logging.info('STARTING EVALUATION')
                test, valid, runtime, params = naseval.eval_model(
                    config=args.__dict__, model=full_vals[0][0])

                index = np.random.choice(list(range(3)))
                test, valid, runtime, params = np.mean(test), np.mean(valid), np.mean(runtime), np.mean(params)
                logging.info('TEST ERROR: %.3f | VALID ERROR: %.3f | RUNTIME: %f | PARAMS: %d'
                                % (test, valid, runtime, params))
            else:
                best_rounds.append(sample_vals[0])

            # Save the fully evaluated architectures
            with open(os.path.join(self.save_dir,
                                   'full_val_architecture_epoch_{}.obj'.format(self.model.epochs)), 'wb') as f:
                pickle.dump(full_vals, f)
        return best_rounds


if __name__ == "__main__":
    if args.search_space == '1':
        search_space = SearchSpace1()
    elif args.search_space == '2':
        search_space = SearchSpace2()
    elif args.search_space == '3':
        search_space = SearchSpace3()
    else:
        raise ValueError('Unknown search space')

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info(args)

    if args.benchmark == 'ptb':
        raise ValueError('PTB not supported.')
    else:
        data_size = 25000
        time_steps = 1

    B = int(args.epochs * data_size / args.batch_size / time_steps)
    if args.benchmark == 'cnn':
        from optimizers.random_search_with_weight_sharing.darts_wrapper_discrete import DartsWrapper
        model = DartsWrapper(save_dir, args.seed, args.batch_size, args.grad_clip, args.epochs, gpu=args.gpu,
                             num_intermediate_nodes=search_space.num_intermediate_nodes, search_space=search_space,
                             init_channels=args.init_channels, cutout=args.cutout)
    else:
        raise ValueError('Benchmarks other cnn on cifar are not available')

    searcher = Random_NAS(B, model, args.seed, save_dir)
    logging.info('budget: %d' % (searcher.B))
    if not args.eval_only:
        searcher.run()
        # archs = searcher.get_eval_arch()
    else:
        np.random.seed(args.seed + 1)
        archs = searcher.get_eval_arch(2)
    # logging.info(archs)
    # arch = ' '.join([str(a) for a in archs[0][0]])
    # with open('/tmp/arch', 'w') as f:
    #     f.write(arch)
