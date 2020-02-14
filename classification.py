import numpy as np
import tree_utils as tu
from node import Node
from eval import Evaluator
from draw_tree import DrawTree


class DecisionTreeClassifier(object):
    """
    A decision tree classifier

    Attributes
    -----------
    is_trained : bool
        Keeps track of whether the classifier has been trained
    root: Node
        Stores the tree structure
    max_depth: int
        (hyper-parameter) Limits maximum depth of the tree

    Methods
    -------
    train(X, y)
        Constructs a decision tree from data X and label y
    predict(X)
        Predicts the class label of samples X
    visualise(indentation)
        Textual visualisation of tree with indentation representing number of spaces
    visualise2d(spacing)
        Creates a plot of the Decision Tree
    chi_prune(tolerance=0.05)
        Prunes the tree using chi squared pruning
        Uses features in x and labels y to prune the tree (Validation)
    prune(X, y_true)
        Prunes the tree, by removing leaf nodes which do not decrease accuracy
        for data X, labels y_true (use Validation set)
    get_accuracy(X, y_true)
        Predicts labels for data X, returns accuracy of prediction for y_true
    """

    att_name = {0: 'x-box', 1: 'y-box', 2: 'width', 3: 'high', 4: 'onpix', 5: 'x-bar', 6: 'y-bar', 7: ' x2bar',
                8: 'y2bar', 9: 'xybar', 10: 'x2ybr', 11: 'xy2br', 12: 'x-ege', 13: 'xegvy', 14: 'y-ege', 15: 'yegvx'}

    def __init__(self, max_depth=100):
        self.is_trained = False
        self.root = None
        self.max_depth = max_depth

    def train(self, x, y):
        """ Constructs a decision tree classifier from data

        :param x: numpy.array
            An N by K numpy array (N is the number of instances, K is the
            number of attributes)
        :param y: numpy.array
            An N-dimensional numpy array

        :returns DecisionTreeClassifier
            A copy of the DecisionTreeClassifier instance
        """

        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."

        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################

        self.root = tu.induce_decision_tree(x, y, self.max_depth)

        self.is_trained = True

        return self

    def predict(self, x):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.

        Assumes that the DecisionTreeClassifier has already been trained.

        :param x: 2D numpy.array
            An N by K numpy array (N is the number of samples, K is the
            number of attributes)

        :returns numpy.array
            An N-dimensional numpy array containing the predicted class label
            for each instance in x
        """
        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################

        # make sure that classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("Decision Tree classifier has not yet been trained.")

        predictions = np.array([tu.classify_sample(sample, self.root) for sample in x])
        return predictions

    def visualise(self, indent=5, node=None, limit=None):
        """ Textual Visualisation of the Decision Tree

        Assumes that the DecisionTreeClassifier has already been trained.

        :param indent: int
            number of spaces for indentation
        :param node:
            root node of the tree will be used
        :param limit:
            maximal depth reached when printing
        """
        if not self.is_trained:
            raise Exception("Decision Tree classifier has not yet been trained.")
        # Root Case
        if node is None:
            node = self.root
        self._visualise(indent, node, limit, 0)

    def _visualise(self, indent, node, limit, current_depth):
        if limit is not None and current_depth > limit:
            return

        # Update space indentation
        space = indent * node.depth
        # Check if leaf, then return, otherwise print node data in full:
        if node.isLeaf:
            print(space*' ' + '+---Leaf', node.label, 'with labels: ' + str(node.label_dict))
            return
        else:
            print(space * ' ' + '+---IntNode', str(node.attribute) + ':' +
                  DecisionTreeClassifier.att_name[node.attribute] + " >", node.value,
                  'with entropy', '%.3f' % node.entropy, 'and labels: ' + str(node.label_dict))
            for next_node in node.children_nodes:
                self._visualise(indent, next_node, limit, current_depth+1)

    def visualise_2d(self, spacing=5):
        draw_tree = DrawTree(self.root)
        draw_tree.print_tree(spacing)

    def chi_prune(self, tolerance=0.05):
        tu.chi_prune(tolerance, self.root)

    def get_depth(self):
        return tu.get_depth(self.root)

    def prune(self, x, y_true):
        """ Prunes the trained tree by recursively removing leaf nodes
                which do not increase accuracy

        :param x: np.array
            data that will be used for predictions
        :param y_true:
            true labels for data in x
        """
        self.best_accuracy = self.get_accuracy(x, y_true)
        self._prune(x, y_true, self.root)

    def _prune(self, x, y_true, node):

        # reached leaf node
        if node.isLeaf is True:
            return Node

        child1 = self._prune(x, y_true, node.children_nodes[0])
        child2 = self._prune(x, y_true, node.children_nodes[1])
        if child1 is not None and child2 is not None:
            node.isLeaf = True
            y_predicted = self.predict(x)
            pruned_acc = Evaluator.get_accuracy(y_predicted, y_true)
            if pruned_acc >= self.best_accuracy:
                self.best_accuracy = pruned_acc
                node.children_nodes.clear()
                node.attribute = None
                node.value = None
                return Node
            else:
                node.isLeaf = False

        return None

    def get_accuracy(self, x, y_true):
        y_predicted = self.predict(x)
        return Evaluator.get_accuracy(y_predicted, y_true)




