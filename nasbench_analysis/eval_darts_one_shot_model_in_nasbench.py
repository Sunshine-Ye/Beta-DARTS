import glob
import json
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from nasbench import api

from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2
from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3
from nasbench_analysis.utils import get_top_k, INPUT, OUTPUT, CONV1X1, NasbenchWrapper, natural_keys
from optimizers.darts.genotypes import PRIMITIVES


nasbench = NasbenchWrapper(
    dataset_file='../../nasbench_only108.tfrecord')


def softmax(weights, axis=-1):
    return F.softmax(torch.Tensor(weights), axis).data.cpu().numpy()


def get_directory_list(path):
    """Find directory containing config.json files"""
    directory_list = []
    # return nothing if path is a file
    if os.path.isfile(path):
        return []
    # add dir to directorylist if it contains .json files
    if len([f for f in os.listdir(path) if f == 'config.json']) > 0:
        directory_list.append(path)
    for d in os.listdir(path):
        new_path = os.path.join(path, d)
        if os.path.isdir(new_path):
            directory_list += get_directory_list(new_path)
    return directory_list


def eval_one_shot_model(config, model):
    model_list = pickle.load(open(model, 'rb'))

    alphas_mixed_op = model_list[0]
    chosen_node_ops = softmax(alphas_mixed_op, axis=-1).argmax(-1)

    node_list = [PRIMITIVES[i] for i in chosen_node_ops]
    alphas_output = model_list[1]
    alphas_inputs = model_list[2:]

    if config['search_space'] == '1':
        search_space = SearchSpace1()
        num_inputs = list(search_space.num_parents_per_node.values())[3:-1]
        parents_node_3, parents_node_4 = \
            [get_top_k(softmax(alpha, axis=1), num_input) for num_input, alpha in zip(num_inputs, alphas_inputs)]
        output_parents = get_top_k(softmax(alphas_output), num_inputs[-1])
        parents = {
            '0': [],
            '1': [0],
            '2': [0, 1],
            '3': parents_node_3,
            '4': parents_node_4,
            '5': output_parents
        }
        node_list = [INPUT, *node_list, CONV1X1, OUTPUT]

    elif config['search_space'] == '2':
        search_space = SearchSpace2()
        num_inputs = list(search_space.num_parents_per_node.values())[2:]
        parents_node_2, parents_node_3, parents_node_4 = \
            [get_top_k(softmax(alpha, axis=1), num_input) for num_input, alpha in zip(num_inputs[:-1], alphas_inputs)]
        output_parents = get_top_k(softmax(alphas_output), num_inputs[-1])
        parents = {
            '0': [],
            '1': [0],
            '2': parents_node_2,
            '3': parents_node_3,
            '4': parents_node_4,
            '5': output_parents
        }
        node_list = [INPUT, *node_list, CONV1X1, OUTPUT]

    elif config['search_space'] == '3':
        search_space = SearchSpace3()
        num_inputs = list(search_space.num_parents_per_node.values())[2:]
        parents_node_2, parents_node_3, parents_node_4, parents_node_5 = \
            [get_top_k(softmax(alpha, axis=1), num_input) for num_input, alpha in zip(num_inputs[:-1], alphas_inputs)]
        output_parents = get_top_k(softmax(alphas_output), num_inputs[-1])
        parents = {
            '0': [],
            '1': [0],
            '2': parents_node_2,
            '3': parents_node_3,
            '4': parents_node_4,
            '5': parents_node_5,
            '6': output_parents
        }
        node_list = [INPUT, *node_list, OUTPUT]

    else:
        raise ValueError('Unknown search space')

    adjacency_matrix = search_space.create_nasbench_adjacency_matrix(parents)
    # Convert the adjacency matrix in format for nasbench
    adjacency_list = adjacency_matrix.astype(np.int).tolist()
    model_spec = api.ModelSpec(matrix=adjacency_list, ops=node_list)
    # Query nasbench
    data = nasbench.query(model_spec)
    valid_error, test_error, runtime, params = [], [], [], []
    for item in data:
        test_error.append(1 - item['test_accuracy'])
        valid_error.append(1 - item['validation_accuracy'])
        runtime.append(item['training_time'])
        params.append(item['trainable_parameters'])
    return test_error, valid_error, runtime, params


def eval_directory(path):
    """Evaluates all one-shot architecture methods in the directory."""
    # Read in config
    with open(os.path.join(path, 'config.json')) as fp:
        config = json.load(fp)
    # Accumulate all one-shot models
    one_shot_architectures = glob.glob(os.path.join(path, 'one_shot_architecture_*.obj'))
    # Sort them by date
    one_shot_architectures.sort(key=natural_keys)
    # Eval all of them
    test_errors = []
    valid_errors = []
    for model in one_shot_architectures:
        test, valid, _, _ = eval_one_shot_model(config=config, model=model)
        test_errors.append(test)
        valid_errors.append(valid)

    with open(os.path.join(path, 'one_shot_validation_errors.obj'), 'wb') as fp:
        pickle.dump(valid_errors, fp)

    with open(os.path.join(path, 'one_shot_test_errors.obj'), 'wb') as fp:
        pickle.dump(test_errors, fp)

