from eval import Evaluator
from classification import DecisionTreeClassifier
import numpy as np


class kTrees(object):

    def __init__(self, models, accuracies):
        self.models = models
        self.accuracies = accuracies

    @staticmethod
    def cross_validation(x, y, k, pruning):
        """
        Performs k-cross validation
        :return: Mean accuracy, Standard Deviation for accuracies and a kTrees object containing the models
        """
        # transform into column vector
        y = y.reshape(-1, 1)
        # join samples and labels --> will change data type of array
        data = np.concatenate((x, y), axis=1)
        # randomly shuffle data
        np.random.shuffle(data)

        # split data into k subsets
        splits = np.array_split(data, k)

        models = []
        accuracy_all = []

        for i_test in range(k):
            # create training set with all sets apart from evaluation set
            training_set = np.concatenate([splits[i] for i in range(k) if i != i_test])
            # select all but last column (i.e. get samples) --> need to change type
            x_train = training_set[:, :-1].astype(int)
            # select last column only (i.e. get labels) --> need to change type
            y_train = training_set[:, -1].astype(np.str_)

            tree = DecisionTreeClassifier()
            tree.train(x_train, y_train)

            if pruning:
                tree.chi_prune()

            models.append(tree)

            # test model on the hold out Test set
            x_test = splits[i_test][:, :-1].astype(int)
            y_test = splits[i_test][:, -1].astype(np.str_)
            y_predicted = tree.predict(x_test)
            accuracy = Evaluator.get_accuracy(y_predicted, y_test)
            accuracy_all.append(accuracy)

        k_forest = kTrees(models, accuracy_all)
        mean_accuracy = np.average(accuracy_all)
        sd = np.std(accuracy_all)
        return mean_accuracy, sd, k_forest

    def get_best_model(self):
        best_index = self.accuracies.index(max(self.accuracies))
        return self.models[best_index]

    def predict(self, x):
        labels_total = x.shape[0]
        models_total = len(self.models)
        all_predictions = np.zeros((models_total, labels_total), dtype=np.str_)

        for i in range(models_total):
            predictions = self.models[i].predict(x)
            all_predictions[i] = predictions

        predictions = np.zeros(labels_total, dtype=np.str_)
        for i in range(labels_total):
            unique, counts = np.unique(all_predictions[:, i], return_counts=True)
            most_predicted_index = np.where(counts == max(counts))[0][0]
            predictions[i] = unique[most_predicted_index]

        return predictions
