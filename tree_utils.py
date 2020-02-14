"""
    Helper functions for DecisionTreeClassification class

        get_entropy(labels)
            Returns entropy for labels provided

        get_average_entropy(children_labels)
            Returns weighted entropy for a list of labels

        information gain(entropy_parent, children_labels)
            Returns information gain based on entropy of parent children labels

        sort_by_attribute(x, y, attribute
            Sorts features in x and labels in y, by attribute in ascending order

        split_at_index(array, split_index)
            Splits array at specified index (row)

        split_samples_by_attribute_value(x, y, attribute, split_value)
            Sort data by attribute and split at specified value

        get_optimal_split(x, y)
            Finds split (attribute and value) which leads to most information gain

        induce_decision_tree(x, y, max_depth, cur_depth=0)
            Creates a tree structure of Nodes for classification of features x with
                associated labels y

        classify_sample(sample, node)
            Predicts the label for sample using given node (root node)

        k_child_calculator(dictA, dictB):
            Calculates k value (used in chi_prunning) from frequency of labels in A and B

        chi_prune(tolerance, node)
            Prunes all nodes (given root) using chi pruning

        get_depth(node)
            Returns maximum depth of node
"""

from scipy.stats import chi2
from node import Node
import numpy as np


def get_entropy(labels):
    """ Calculates entropy for labels

    :param labels: np.array
        1D numpy array of labels
    :return: int
        current entropy for labels
    """
    (_, label_counts) = np.unique(labels, return_counts=True)
    label_probabilities = np.array(label_counts / labels.shape[0])
    return -sum(label_probabilities * np.log2(label_probabilities))


def get_average_entropy(children_labels):
    """ Returns weighted entropy for a list of labels

    :param children_labels: list of 1D np.arrays
    :return: int
    """
    total = sum(subset.shape[0] for subset in children_labels)

    weighted_entropies = [subset.shape[0] / total *
                          get_entropy(subset) for subset in children_labels]
    return sum(weighted_entropies)


def information_gain(entropy_parent, children_labels):
    """ Calculates change in entropy between parent entropy and
            entropy of children_labels

    :param entropy_parent: int
    :param children_labels: list of 1D np.arrays
    :return: int
    """
    return entropy_parent - get_average_entropy(children_labels)


def sort_by_attribute(x, y, attribute):
    """ Sorts features in x and labels in y, by attribute in ascending order

    :param x: 2D np.array
        features (attributes)
    :param y: 1D np.array
        labels for x
    :param attribute: int
        attribute (column) to sort by
    :return: (sorted_x, sorted_y)
        sorted by attribute in ascending order
    """
    sorted_idx = x[:, attribute].argsort()
    sorted_x = x[sorted_idx]
    sorted_y = y[sorted_idx]
    return sorted_x, sorted_y


def split_at_index(array, split_index):
    """ Splits array at specified index (row)

    :param array: np.array (1D or 2D)
    :param split_index: int
    :return: list of np.arrays
    """
    children_arrays = [array[0:split_index], array[split_index:]]
    return children_arrays


def split_samples_by_attribute_value(x, y, attribute, split_value):
    """ Sorts x and y by given attribute of x. Finds split index (i.e. where
            sorted feature value > split_value). Splits x and y at
            split index. Returns children_x and children_y

    Note: x and y will be sorted upon return

    :param x: 2D np.array
        features
    :param y: 1D np.array
        labels for x
    :param attribute: int
        attribute (column) to sort by
    :param split_value: int
        value of attribute at which to split
    :return: (list of np.arrays, list of np.arrays)
    """
    x, y = sort_by_attribute(x, y, attribute)

    split_index = x.shape[0]
    for i in range(split_index):
        if x[i][attribute] > split_value:
            split_index = i
            break

    children_x = split_at_index(x, split_index)
    children_y = split_at_index(y, split_index)
    return children_x, children_y


def get_optimal_split(x, y):
    """ Finds split (attribute and value) which leads to most information gain

    Note: returns None if no information gain split is possible

    :param x: 2D np.array
        features
    :param y: 1D np.array
        labels for x
    :return: (int, int) or None
        (attribute, value)
        attribute - column of x to sort by
        value - value to use for the split
    """
    parent_entropy = get_entropy(y)
    split_tuple = None
    best_gain = 0

    for attribute in range(x.shape[1]):
        x, y = sort_by_attribute(x, y, attribute)
        limit = x.shape[0] - 1
        for index in range(limit):
            if x[index][attribute] != x[index + 1][attribute]:
                children_y = split_at_index(y, index + 1)
                gain = information_gain(parent_entropy, children_y)
                if gain > best_gain:
                    split_tuple = (attribute, x[index][attribute])
                    best_gain = gain
    return split_tuple


def induce_decision_tree(x, y, max_depth, cur_depth=0):
    """ Creates a tree structure of Nodes for classification of features x with
            associated labels y

    :param x: 2D np.array
        features
    :param y: 1D np.array
        labels for X
    :param max_depth: int
        maximum depth for decision tree
    :param cur_depth: 0
        current depth of node
    :return: Node
        root node of Decision Tree
    """
    split_tuple = get_optimal_split(x, y)
    if split_tuple is None or \
            cur_depth >= max_depth - 1:
        return Node(cur_depth, y)

    attribute, value = split_tuple
    node = Node(cur_depth, y, attribute, value)

    children_x, children_y = split_samples_by_attribute_value(x, y, attribute, value)

    for i in range(len(children_x)):
        x = children_x[i]
        y = children_y[i]
        # recursion is here
        node.add_child(induce_decision_tree(x, y, max_depth, cur_depth + 1))

    return node


def classify_sample(sample, node):
    """ Predicts the label for sample using given node (root node)

    :param sample: 1D np.array
        features for single element
    :param node: root node
    :return: char
        predicted label for given sample
    """
    if node.isLeaf is True:
        return node.label
    if sample[node.attribute] > node.value:
        return classify_sample(sample, node.children_nodes[1])
    return classify_sample(sample, node.children_nodes[0])


def k_child_calculator(dictA, dictB):
    """ Calculates k value (used in chi_prunning) from frequency of labels in A and B

    :param dictA: dictionary of label:count
    :param dictB: dictionary of label:count
    :return: int
    """
    proportion = sum(dictB.values())/sum(dictA.values())
    k = 0
    for label in dictA.keys():
        expected_in_B = dictA[label]*proportion
        if label in dictB.keys():
            actual_in_B = dictB[label]
        else:
            actual_in_B = 0
        k += (expected_in_B-actual_in_B)**2/expected_in_B
    return k


def chi_prune(tolerance, node):
    """ Prunes all nodes (given root) using chi pruning

    :param tolerance: int
    :param node: Node
        root node
    :return: Node (root)
        pruned
    """
    if node.isLeaf:
        return node
    else:
        k = 0
        df = len(node.label_dict.keys())*len(node.children_nodes) - 1
        for child in node.children_nodes:
            child = chi_prune(tolerance, child)
            if child is not None:
                k += k_child_calculator(node.label_dict, child.label_dict)

        if k <= chi2.isf(tolerance, df):
            node.children_nodes.clear()
            node.attribute = None
            node.value = None
            node.isLeaf = True
        return node


def get_depth(node, max_depth=0):
    """ Returns maximum depth of node

    :param node: Node
    :return: int
    """
    max_depth = max(max_depth, node.depth)
    if not node.isLeaf:
        for child_node in node.children_nodes:
            depth = get_depth(child_node, max_depth)
            max_depth = max(max_depth, depth)

    return max_depth

