## CO395 Introduction to Machine Learning: Coursework 1 (Decision Trees)

### Introduction

This repository contains code for training and using a Decision Tree, as well
as code for interacting with data and evaluate perfomance.

### Data

The ``data/`` directory contains the datasets that can be used with Decision
Tree Classifier.

The primary datasets are:
- ``train_full.txt``
- ``train_sub.txt``
- ``train_noisy.txt``
- ``validation.txt``

Some simpler datasets that you may use to help you with implementation or
debugging:
- ``toy.txt``
- ``simple1.txt``
- ``simple2.txt``

The official test set is ``test.txt``. Please use this dataset sparingly and
purely to report the results of evaluation. Do not use this to optimise your
classifier (use ``validation.txt`` for this instead).


### Codes

- ``dataset.py``

    * Class to load data from Data directory

- ``classification.py``

	* Contains the code for the ``DecisionTreeClassifier``.

- ``eval.py``

	* Contains the code for the ``Evaluator`` class. Contains
``confusion_matrix()``, ``accuracy()``, ``precision()``,  ``recall()``,
and ``f1_score()`` methods.

- ``k_trees.py``

    * Contains code for training multiple tree and conducting cross-validation

- ``example_main.py``

	* Contains an example of how the evaluation script on LabTS might use the classes
and invoke the methods defined in ``classification.py`` and ``eval.py``.


### Instructions

#### Loading the data

For ease of interaction with data use the ``Dataset class`` in ``dataset.py``.
- To load data from file:
    ```python
      from dataset import Dataset
      dataset = Dataset(pathToFile.txt)
    ```
- To get attributes and labels:
    ```python
       attributes = dateset.attributes
       labels = dataset.labels
    ```
- To visualise data:
    ```python
      dataset.range_graph()
      dataset.histogram()
    ```

#### Training a Decision Tree

All relevant methods for training and using the Decision Tree are part of the ``DecisionTreeClassifier``
class located in ``classification.py``.

- Instantiating
    - An optional ``max_depth`` hyper-parameter can be passed into the constructor to limit
    maximal depth of the tree  (set by default to 100)

  ```python
    from classification import DecisionTreeClassifier  
    tree = DecisionTreeClassifier()
  ```

- Training Decision Tree
    ```python
  dataset = Dataset(pathToFile.txt)
  tree = DecisionTreeClassifier()

  x = dataset.attributes
  y = dataset.labels
  tree.train(x, y)
     ```

- Predicting labels (assuming tree has been trained)
    ```python
    testDataset = Dataset(pathToTest.txt)
    x = testDataset.attributes

    y = tree.predict(x_features)
    ```

- Pruning the tree (assume tree has been trained)
    - Chi pruning can be used, optional tolerance argument can be passed
    ```python
    tree.chi_prune()
    ```
    - Reduced Error Pruning can be used, requires validation data
    ```python
    validation = Dataset(pathToValidation.txt)
    x = validation.attributes
    y = validation.labels

    tree.prune(x, y)
    ```

- Text visualisation of tree: ``tree.visualise()``
- Graphical visualisation of tree: ``tree.visualise2d()`` or ``Buchheim`` object

#### Evaluation
Evaluation of perfomance (i.e. predicted vs true labels) can be done using the ``Evaluator``
class located in ``eval.py``
- Creating a confusion matrix
    - Optional ``class_labels`` argument can be passed to determine
    order of labels in the confusion matrix
    ```python
  from eval.py import Evaluator
  confusion = Evaluator.confusion_matrix(predictions, ground_truth, class_labels)
  ```
- Metrics: accuracy, precision, recall, F1
    - All require confusion metrics as an argument
    - Accuracy returns an int
    - Precision, Recall and F1 all return tuple, with first element
    being an array of per class perfomance, and second element being the
    macro-average
    ```python
    accuracy = Evaluator.accuracy(confusion)

  p, macro_p = Evaluator.precision(confusion)
  r, macro_r = Evaluator.recall(confusion)
  f, macro_f = Evaluator.f1_score(confusion)
    ```

#### K Cross Validation and MultiTree Predictions
For both cross validation and using multiple trees for prediction, the
class ``kTrees`` from ``k_trees.py`` can be used.
A call to kTrees.cross_validation() will require the samples (x), the true
labels (y), a value for k and a boolean (pruning) to specify if chi pruning
should be executed or not.

- Cross validation metrics
    ```python
    from k_trees import kTrees
    from dataset import Dataset
    data = Dataset("data/train_full.txt")
    x = data.attributes
    y = data.labels
    k = 10
    average_accuracy, standard_deviation, _ = kTrees.cross_validation(x, y, k, pruning=False)
  ```

- Multi Model prediction by majority vote
    ```python
    from k_trees import kTrees
    from dataset import Dataset
    data = Dataset("data/train_full.txt")
    x = data.attributes
    y = data.labels
    k = 10
    _, _, ktrees = kTrees.cross_validation(x, y, k, pruning=False)

    testData = Dataset(pathToTest.txt)
    x_test = testData.attributes
    y_predicted = ktrees.predict(x_test)
```
