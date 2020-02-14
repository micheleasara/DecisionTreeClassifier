import matplotlib.pyplot as plt
from classification import DecisionTreeClassifier
from dataset import Dataset


class Buchheim(object):
    def __init__(self, root_node, parent=None, number=1):
        self.root = root_node
        self.x = -1
        self.y = root_node.depth
        self.parent = parent
        self.children = [Buchheim(child, self, i + 1)
                         for i, child
                         in enumerate(root_node.children_nodes)]
        self.thread = None
        self.mod = 0
        self.ancestor = self
        self.change = self.shift = 0
        self._lmost_sibling = None
        self.number = number

    def left(self):
        """
        Function to return the left thread of a tree
        :return: left thread or other if none
        """
        return self.thread or len(self.children) and self.children[0]

    def right(self):
        """
        Function to return the right thread of a tree
        :return: right thread or other if none
        """
        return self.thread or len(self.children) and self.children[-1]


    def left_brother(self):
        """
        Function to return the left brother of this tree node
        :return: Buchheim tree
        """
        n = None
        if self.parent:
            for node in self.parent.children:
                if node == self:
                    return n
                else:
                    n = node
        return n

    def get_lmost_sibling(self):
        """
        Function to that sets/returns the _lmost_sibling attribute
        :return: None or a Buchheim tree
        """
        if not self._lmost_sibling and self.parent and self != self.parent.children[0]:
            self._lmost_sibling = self.parent.children[0]
        return self._lmost_sibling

    lmost_sibling = property(get_lmost_sibling)

    def print_tree(self, lim):
        """
        Function that creates a figure and calls the plot_tree function
        :param lim: int
        :return:
        """
        plt.figure()
        self.plot_tree(lim)
        plt.show()

    def plot_tree(self, lim):
        """
        Function that recursively plots this tree
        :param lim: int
        :return:
        """
        # Plot relevant lines to children nodes (if any)
        length = len(str(self.root))
        if length > 1:
            font = 6
            op = 0.5
            diff = 0.1
        else:
            font = 8
            op = 1
            diff = -0.1
        for child in self.children:
            if child.y <= lim:
                child.plot_tree(lim)
                plt.plot([self.x, child.x],
                         [-self.y, -child.y])
        # Plot this node as a point:
        plt.plot(self.x, -self.y, 'k')
        # Plot this node as a box with info:

        plt.annotate(self.root, (self.x, -self.y + diff),
                     bbox=dict(facecolor='sienna', alpha=op, boxstyle='circle'),
                     fontsize=font, ha='center')
        return


def buchheim(tree):
    """
    Function that calls the buchheim utility functions below to run through the first/second/third walks
    to set up the tree in full and determine the appropriate x values.
    :param tree: Buchheim tree to set up
    :return: Buchheim tree, set up
    """
    dt = firstwalk(tree)
    min = second_walk(dt)
    if min < 0:
        third_walk(dt, -min)
    return dt

def firstwalk(v, distance=1.):
    """
    Function to complete the first walk of the tree to set initial x-values for nodes.
    :param v: Buchheim tree
    :param distance: float
    :return: A Buchheim tree
    """
    if len(v.children) == 0:
        if v.lmost_sibling:
            v.x = v.left_brother().x + distance
        else:
            v.x = 0.

    else:
        default_ancestor = v.children[0]
        for w in v.children:
            firstwalk(w, distance)
            default_ancestor = apportion(w, default_ancestor, distance)
        execute_shifts(v)

        midpoint = (v.children[0].x + v.children[-1].x) / 2

        w = v.left_brother()
        if w:
            v.x = w.x + distance
            v.mod = v.x - midpoint
        else:
            v.x = midpoint

    return v

def second_walk(v, m=0, depth=0, minimum=None):
    """
    Function to perform a second walk of the Buchheim tree, in order to determine the minimum amount needed to shift
    nodes by to avoid clashes/implement the mod shifts too - allows the process to be linear.
    :param v: Buchheim tree
    :param m: int
    :param depth: int
    :param minimum: None or int
    :return: int
    """
    v.x += m
    v.y = depth
    if minimum is None or v.x < minimum:
        minimum = v.x

    for w in v.children:
        minimum = second_walk(w, m + v.mod, depth+1, minimum)

    return minimum

def third_walk(tree, n):
    """

    :param tree:
    :param n:
    :return:
    """
    tree.root.x += n
    for child in tree.children:
        third_walk(child, n)


def apportion(v, default_ancestor, distance):
    """
    Function to carry out apportioning of tree!
    :param v: Buchheim tree
    :param default_ancestor: Buchheim tree
    :param distance: int
    :return: Buchheim tree
    """
    w = v.left_brother()
    if w is not None:
        #The following are used as per Buchheim notation:
        #i = inner; o = outer; r = right; l = left; r is positive; l us negative
        vir = vor = v
        vil = w
        vol = v.lmost_sibling
        sir = sor = v.mod
        sil = vil.mod
        sol = vol.mod
        while vil.right() and vir.left():
            vil = vil.right()
            vir = vir.left()
            vol = vol.left()
            vor = vor.right()
            vor.ancestor = v
            shift = (vil.x + sil) - (vir.x + sir) + distance
            if shift > 0:
                move_subtree(ancestor(vil, v, default_ancestor), v, shift)
                sir = sir + shift
                sor = sor + shift
            sil += vil.mod
            sir += vir.mod
            sol += vol.mod
            sor += vor.mod
        if vil.right() and not vor.right():
            vor.thread = vil.right()
            vor.mod += sil - sor
        else:
            if vir.left() and not vol.left():
                vol.thread = vir.left()
                vol.mod += sir - sol
            default_ancestor = v

    return default_ancestor

def move_subtree(wl, wr, shift):
    """
    Function to move a subtree a particular amount
    :param wl: Buchheim tree
    :param wr: Buchheim tree
    :param shift: int
    :return:
    """
    subtrees = wr.number - wl.number
    wr.change -= shift / subtrees
    wr.shift += shift
    wl.change += shift / subtrees
    wr.x += shift
    wr.mod += shift

def execute_shifts(v):
    """
    Function execute the shifts required to avoid clashes
    :param v: Buchheim tree
    :return:
    """
    shift = change = 0
    for w in v.children[::-1]:
        w.x += shift
        w.mod += shift
        change += w.change
        shift += w.shift + change

def ancestor(vil, v, default_ancestor):
    """
    Function that returns the default ancestor of a tree
    :param vil: Buchheim tree
    :param v: Buchheim tree
    :param default_ancestor: Buchheim tree
    :return: Buchheim tree
    """
    if vil.ancestor in v.parent.children:
        return vil.ancestor
    else:
        return default_ancestor

def example_model(train_filename, lim):
    """
    Example function that trains a tree and implements the above to visualise it
    :param train_filename: filename for training
    :param lim: int
    :return:
    """
    training_dataset = Dataset(train_filename)
    x = training_dataset.attributes
    y = training_dataset.labels

    # Training on whole dataset
    print("Training the decision tree...")
    tree = DecisionTreeClassifier()
    tree = tree.train(x, y)
    tree.chi_prune()
    #tree.visualise_2d(15)
    drawtree = Buchheim(tree.root)
    bt = buchheim(drawtree)
    bt.print_tree(lim)


if __name__ == "__main__":
    TRAINING_FILES = ["data/train_full.txt", "data/train_noisy.txt", "data/train_sub.txt",
                      "data/simple1.txt", "data/simple2.txt", "data/toy.txt"]
    VALIDATION_FILE = "data/validation.txt"
    TEST_FILE = "data/test.txt"

    example_model(TRAINING_FILES[0], 6)

    # cross_validation(TRAINING_FILES[0])

    # multi_model_predictions(TRAINING_FILES[0], TEST_FILE)