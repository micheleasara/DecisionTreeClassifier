import numpy as np
import tree_utils


class Node(object):
    """
        Class to define the nodes of a decision tree

        @attributes:
            depth: depth of node in the tree (0-index based)
            isLeaf: set to True if node is a leaf node
            label: represents the majority label at node
            entropy: represents entropy at node

            If isLeaf is False:
                attribute: attribute to split by
                value: split at > value
                children_nodes: list of children nodes
        """

    def __init__(self, depth=None, labels=None, attribute=None, value=None):
        """
            Depth and labels passed to create leaf node
            Depth, labels, attribute and value passed to create split node
        """
        # Plotting attributes
        self.depth = depth
        self.x = 0
        # Label Attributes
        self.label, self.label_dict = Node.get_majority_label(labels)
        self.entropy = tree_utils.get_entropy(labels)
        self.attribute = attribute
        self.value = value
        # Children Attributes
        self.children_nodes = []
        # Branch/Leaf Boolean Attribute:
        if attribute is None:
            self.isLeaf = True
        else:
            self.isLeaf = False

    def add_child(self, node):
        self.children_nodes.append(node)

    @staticmethod
    def get_majority_label(labels):
        (unique_labels, label_counts) = np.unique(labels, return_counts=True)
        # TODO maybe deal with clashes when classes have equal occurrences
        label_dict = {}
        for i in range(0,len(label_counts)):
            label_dict[unique_labels[i]] = label_counts[i]
        index_most_common = np.where(label_counts == max(label_counts))[0][0]
        return unique_labels[index_most_common], label_dict

    def __repr__(self):
        if self.isLeaf is True:
            return "%s" % self.label
        else:
            rule = '\n'.join((
                r'IntNode %g' % self.attribute,
                r'Rule: <= %g' % self.value))
        return rule






