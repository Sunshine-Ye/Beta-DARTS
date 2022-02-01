import itertools
import os
import re

import networkx as nx
import numpy as np
import seaborn as sns
from nasbench import api

sns.set_style('whitegrid')

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OUTPUT_NODE = 6


class NasbenchWrapper(api.NASBench):
    """Small modification to the NASBench class, to return all three architecture evaluations at
    the same time, instead of samples."""

    def query(self, model_spec, epochs=108, stop_halfway=False):
        """Fetch one of the evaluations for this model spec.

        Each call will sample one of the config['num_repeats'] evaluations of the
        model. This means that repeated queries of the same model (or isomorphic
        models) may return identical metrics.

        This function will increment the budget counters for benchmarking purposes.
        See self.training_time_spent, and self.total_epochs_spent.

        This function also allows querying the evaluation metrics at the halfway
        point of training using stop_halfway. Using this option will increment the
        budget counters only up to the halfway point.

        Args:
          model_spec: ModelSpec object.
          epochs: number of epochs trained. Must be one of the evaluated number of
            epochs, [4, 12, 36, 108] for the full dataset.
          stop_halfway: if True, returned dict will only contain the training time
            and accuracies at the halfway point of training (num_epochs/2).
            Otherwise, returns the time and accuracies at the end of training
            (num_epochs).

        Returns:
          dict containing the evaluated darts for this object.

        Raises:
          OutOfDomainError: if model_spec or num_epochs is outside the search space.
        """
        if epochs not in self.valid_epochs:
            raise api.OutOfDomainError('invalid number of epochs, must be one of %s'
                                       % self.valid_epochs)

        fixed_stat, computed_stat = self.get_metrics_from_spec(model_spec)
        trainings = []
        for index in range(self.config['num_repeats']):
            computed_stat_at_epoch = computed_stat[epochs][index]

            data = {}
            data['module_adjacency'] = fixed_stat['module_adjacency']
            data['module_operations'] = fixed_stat['module_operations']
            data['trainable_parameters'] = fixed_stat['trainable_parameters']

            if stop_halfway:
                data['training_time'] = computed_stat_at_epoch['halfway_training_time']
                data['train_accuracy'] = computed_stat_at_epoch['halfway_train_accuracy']
                data['validation_accuracy'] = computed_stat_at_epoch['halfway_validation_accuracy']
                data['test_accuracy'] = computed_stat_at_epoch['halfway_test_accuracy']
            else:
                data['training_time'] = computed_stat_at_epoch['final_training_time']
                data['train_accuracy'] = computed_stat_at_epoch['final_train_accuracy']
                data['validation_accuracy'] = computed_stat_at_epoch['final_validation_accuracy']
                data['test_accuracy'] = computed_stat_at_epoch['final_test_accuracy']

            self.training_time_spent += data['training_time']
            if stop_halfway:
                self.total_epochs_spent += epochs // 2
            else:
                self.total_epochs_spent += epochs
            trainings.append(data)

        return trainings


def get_top_k(array, k):
    return list(np.argpartition(array[0], -k)[-k:])


def parent_combinations(adjacency_matrix, node, n_parents=2):
    """Get all possible parent combinations for the current node."""
    if node != 1:
        # Parents can only be nodes which have an index that is lower than the current index,
        # because of the upper triangular adjacency matrix and because the index is also a
        # topological ordering in our case.
        return itertools.combinations(np.argwhere(adjacency_matrix[:node, node] == 0).flatten(),
                                      n_parents)  # (e.g. (0, 1), (0, 2), (1, 2), ...
    else:
        return [[0]]


def draw_graph_to_adjacency_matrix(graph):
    """
    Draws the graph in circular format for easier debugging
    :param graph:
    :return:
    """
    dag = nx.DiGraph(graph)
    nx.draw_circular(dag, with_labels=True)


def upscale_to_nasbench_format(adjacency_matrix):
    """
    The search space uses only 4 intermediate nodes, rather than 5 as used in nasbench
    This method adds a dummy node to the graph which is never used to be compatible with nasbench.
    :param adjacency_matrix:
    :return:
    """
    return np.insert(
        np.insert(adjacency_matrix,
                  5, [0, 0, 0, 0, 0, 0], axis=1),
        5, [0, 0, 0, 0, 0, 0, 0], axis=0)


def parse_log(path):
    f = open(os.path.join(path, 'log.txt'), 'r')
    # Read in the relevant information
    train_accuracies = []
    valid_accuracies = []
    for line in f:
        if 'train_acc' in line:
            train_accuracies.append(line)
        elif 'valid_acc' in line:
            valid_accuracies.append(line)

    valid_error = [[1 - 1 / 100 * float(re.search('valid_acc ([-+]?[0-9]*\.?[0-9]+)', line).group(1))] for line in
                   valid_accuracies]
    train_error = [[1 - 1 / 100 * float(re.search('train_acc ([-+]?[0-9]*\.?[0-9]+)', line).group(1))] for line in
                   train_accuracies]

    return valid_error, train_error

# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]
