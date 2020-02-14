import numpy as np

from classification import DecisionTreeClassifier
from eval import Evaluator
from dataset import Dataset
from k_trees import kTrees


def range_graph(file):
    """ Ranges of attributes in file """
    data = Dataset(file)
    data.range_graph()


def label_difference(file1, file2):
    """ Displays difference between the labels of file1 and file2 """
    data1 = Dataset(file1)
    data2 = Dataset(file2)
    data1.compare(data2)


def train_tree(file_train, prune=False):
    """ Trains tree on data in file, uses chi_pruning to prune if this is enabled """
    data = Dataset(file_train)
    x = data.attributes
    y = data.labels

    # Training on whole dataset
    print("Training the decision tree...")
    tree = DecisionTreeClassifier()
    tree = tree.train(x, y)

    if prune:
        print("Pruning tree...")
        tree.chi_prune()

    return tree


def test_tree(tested_tree, file_test):
    """ Uses tree to predict labels of data in file, evaluates and prints results """
    testData = Dataset(file_test)
    x = testData.attributes
    y_true = testData.labels

    predictions = tested_tree.predict(x)

    print("Predictions: {}".format(predictions))
    classes = np.unique(predictions)

    print("Confusion Matrix: ")
    confusion = Evaluator.confusion_matrix(predictions, y_true)
    print(confusion)

    accuracy = Evaluator.accuracy(confusion)
    print("\nAccuracy: {}".format(accuracy))

    (p, macro_p) = Evaluator.precision(confusion)
    (r, macro_r) = Evaluator.recall(confusion)
    (f, macro_f) = Evaluator.f1_score(confusion)

    print("\nClass: Precision, Recall, F1")
    for (i, (p1, r1, f1)) in enumerate(zip(p, r, f)):
        print("    {}: {:.2f}, {:.2f}, {:.2f}".format(classes[i], p1, r1, f1))

    print("\nMacro-averaged Precision: {:.9f}".format(macro_p))
    print("Macro-averaged Recall: {:.9f}".format(macro_r))
    print("Macro-averaged F1: {:.2f}".format(macro_f))


def cross_validation(file_train, k=10):
    """ Does cross validation using data in file_train for training, prints average accuracy
            and standard deviation """
    print("Finding average accuracy for cross validation")
    training_dataset = Dataset(file_train)
    x = training_dataset.attributes
    y = training_dataset.labels
    average, sd, _ = kTrees.cross_validation(x, y, k, pruning=False)
    print("{0}-fold cross-validation gives mean accuracy={1:.3f} std={2:.3f}".format(k, average, sd))


def best_model_prediction(file_train, file_test, k=10):
    """ Does cross validation using data in file_train for training,
            selects best model, tests best model on data from file_test """
    training_dataset = Dataset(file_train)
    x = training_dataset.attributes
    y = training_dataset.labels
    _, _, forest = kTrees.cross_validation(x, y, k, pruning=False)
    best_model = forest.get_best_model()
    test_tree(best_model, file_test)


def multi_model_predictions(file_train, file_test):
    """ Trains k models using data in file_train, uses these to predict labels
            from file_test selecting the most occcuring prediction for each """
    training_dataset = Dataset(file_train)
    x = training_dataset.attributes
    y = training_dataset.labels

    # train 10 trees using cross-validation
    _, _, forest = kTrees.cross_validation(x, y, k=10, pruning=False)

    test_tree(forest, file_test)


TRAINING_FILES = ["data/train_full.txt", "data/train_noisy.txt", "data/train_sub.txt",
                  "data/simple1.txt", "data/simple2.txt", "data/toy.txt"]
VALIDATION_FILE = "data/validation.txt"
TEST_FILE = "data/test.txt"

if __name__ == '__main__':
    #range_graph(TRAINING_FILES[0])
    #label_difference(TRAINING_FILES[0], TRAINING_FILES[1])

    tree = train_tree(TRAINING_FILES[0])
    tree.visualise_2d()
    test_tree(tree, TEST_FILE)

    #cross_validation(TRAINING_FILES[0])

    #best_model_prediction(TRAINING_FILES[0], TEST_FILE)

    #multi_model_predictions(TRAINING_FILES[0], TEST_FILE)


