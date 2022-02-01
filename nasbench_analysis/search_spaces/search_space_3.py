import itertools

import matplotlib.pyplot as plt
import numpy as np
from nasbench import api

from nasbench_analysis.search_spaces.search_space import SearchSpace
from nasbench_analysis.utils import OUTPUT_NODE, NasbenchWrapper, INPUT, OUTPUT


class SearchSpace3(SearchSpace):
    def __init__(self):
        self.search_space_number = 3
        self.num_intermediate_nodes = 5
        super(SearchSpace3, self).__init__(search_space_number=self.search_space_number,
                                           num_intermediate_nodes=self.num_intermediate_nodes)
        """
        SEARCH SPACE 3
        """
        self.num_parents_per_node = {
            '0': 0,
            '1': 1,
            '2': 1,
            '3': 1,
            '4': 2,
            '5': 2,
            '6': 2
        }
        if sum(self.num_parents_per_node.values()) > 9:
            raise ValueError('Each nasbench cell has at most 9 edges.')

        self.test_min_error = 0.05338543653488159
        self.valid_min_error = 0.04847759008407593

    def create_nasbench_adjacency_matrix(self, parents, **kwargs):
        # Create nasbench compatible adjacency matrix
        adjacency_matrix = self._create_adjacency_matrix(parents, adjacency_matrix=np.zeros([7, 7]), node=OUTPUT_NODE)
        return adjacency_matrix

    def create_nasbench_adjacency_matrix_with_loose_ends(self, parents):
        return self._create_adjacency_matrix_with_loose_ends(parents)

    def generate_adjacency_matrix_without_loose_ends(self):
        for adjacency_matrix in self._generate_adjacency_matrix(adjacency_matrix=np.zeros([7, 7]), node=OUTPUT_NODE):
            yield adjacency_matrix

    def generate_with_loose_ends(self):
        for parent_node_2, parent_node_3, parent_node_4, parent_node_5, output_parents in itertools.product(
                *[itertools.combinations(list(range(int(node))), num_parents) for node, num_parents in
                  self.num_parents_per_node.items()][2:]):
            parents = {
                '0': [],
                '1': [0],
                '2': parent_node_2,
                '3': parent_node_3,
                '4': parent_node_4,
                '5': parent_node_5,
                '6': output_parents
            }
            adjacency_matrix = self.create_nasbench_adjacency_matrix_with_loose_ends(parents)
            yield adjacency_matrix

    def objective_function(self, nasbench, config, budget=108):
        adjacency_matrix, node_list = super(SearchSpace3, self).convert_config_to_nasbench_format(config)
        node_list = [INPUT, *node_list, OUTPUT]
        adjacency_list = adjacency_matrix.astype(np.int).tolist()
        model_spec = api.ModelSpec(matrix=adjacency_list, ops=node_list)
        nasbench_data = nasbench.query(model_spec, epochs=budget)
        return nasbench_data['validation_accuracy'], nasbench_data['training_time']


def analysis():
    search_space_3 = SearchSpace3()
    search_space_3.sample(with_loose_ends=True)

    cs = search_space_3.get_configuration_space()

    # Load NASBench
    nasbench = NasbenchWrapper('../nasbench_data/108_e/nasbench_full.tfrecord')
    search_space_3.objective_function(nasbench, cs.sample_configuration())

    test_error = []
    valid_error = []

    '''
    for i in range(10000):
        adjacency_matrix, node_list = search_space_3.sample_with_loose_ends()
        adjacency_list = adjacency_matrix.astype(np.int).tolist()
        node_list = [INPUT, *node_list, OUTPUT]
        model_spec = api.ModelSpec(matrix=adjacency_list, ops=node_list)
        nasbench.query(model_spec)
    '''
    for adjacency_matrix, ops, model_spec in search_space_3.generate_search_space_without_loose_ends():
        # Query NASBench
        data = nasbench.query(model_spec)
        for item in data:
            test_error.append(1 - item['test_accuracy'])
            valid_error.append(1 - item['validation_accuracy'])

    print('Number of architectures', len(test_error) / len(data))

    plt.figure()
    plt.title(
        'Distribution of test error in search space (no. architectures {})'.format(int(len(test_error) / len(data))))
    plt.hist(test_error, bins=800)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlabel('Test error')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.xlim(0, 0.3)
    plt.savefig('nasbench_analysis/search_spaces/export/search_space_3/test_error_distribution.pdf', dpi=600)
    plt.show()

    plt.figure()
    plt.title('Distribution of validation error in search space (no. architectures {})'.format(
        int(len(valid_error) / len(data))))
    plt.hist(valid_error, bins=800)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlabel('Validation error')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.xlim(0, 0.3)
    plt.savefig('nasbench_analysis/search_spaces/export/search_space_3/valid_error_distribution.pdf', dpi=600)
    plt.show()

    print('min test_error', min(test_error), 'min valid_error', min(valid_error))


if __name__ == '__main__':
    analysis()
