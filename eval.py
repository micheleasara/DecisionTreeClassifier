import numpy as np


class Evaluator(object):
    """ Class to perform evaluation
    """

    @staticmethod
    def confusion_matrix(prediction, annotation, class_labels=None):
        """ Computes the confusion matrix.
        
        Parameters
        ----------
        prediction : np.array
            an N dimensional numpy array containing the predicted
            class labels
        annotation : np.array
            an N dimensional numpy array containing the ground truth
            class labels
        class_labels : np.array
            a C dimensional numpy array containing the ordered set of class
            labels. If not provided, defaults to all unique values in
            annotation.
        
        Returns
        -------
        np.array
            a C by C matrix, where C is the number of classes.
            Classes should be ordered by class_labels.
            Rows are ground truth per class, columns are predictions.
        """

        if not class_labels:
            class_labels = np.unique(annotation)

        confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

        #######################################################################
        #                 ** TASK 3.1: COMPLETE THIS METHOD **
        #######################################################################

        for i in range(annotation.shape[0]):
            row = np.where(class_labels == annotation[i])[0][0]
            column = np.where(class_labels == prediction[i])[0][0]
            confusion[row][column] += 1

        return confusion

    @staticmethod
    def accuracy(confusion):
        """ Computes the accuracy given a confusion matrix.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions
        
        Returns
        -------
        float
            The accuracy (between 0.0 to 1.0 inclusive)
        """

        #######################################################################
        #                 ** TASK 3.2: COMPLETE THIS METHOD **
        #######################################################################
        total = np.sum(confusion)
        n = confusion.shape[0]
        true_positives = 0
        for i in range(n):
            true_positives += confusion[i][i]

        accuracy = true_positives / total

        return accuracy

    @staticmethod
    def precision(confusion):
        """ Computes the precision score per class given a confusion matrix.
        
        Also returns the macro-averaged precision across classes.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.
        
        Returns
        -------
        np.array
            A C-dimensional numpy array, with the precision score for each
            class in the same order as given in the confusion matrix.
        float
            The macro-averaged precision score across C classes.   
        """

        # Initialise array to store precision for C classes
        p = np.zeros((len(confusion),))

        #######################################################################
        #                 ** TASK 3.3: COMPLETE THIS METHOD **
        #######################################################################

        n = confusion.shape[0]
        for i in range(n):
            true_positives = confusion[i][i]
            p[i] = true_positives / np.sum(confusion[:, i])

        macro_p = np.sum(p) / n

        return p, macro_p

    @staticmethod
    def recall(confusion):
        """ Computes the recall score per class given a confusion matrix.
        
        Also returns the macro-averaged recall across classes.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.
        
        Returns
        -------
        np.array
            A C-dimensional numpy array, with the recall score for each
            class in the same order as given in the confusion matrix.
        
        float
            The macro-averaged recall score across C classes.   
        """

        # Initialise array to store recall for C classes
        r = np.zeros((len(confusion),))

        #######################################################################
        #                 ** TASK 3.4: COMPLETE THIS METHOD **
        #######################################################################

        N = confusion.shape[0]
        for i in range(N):
            true_positives = confusion[i][i]
            r[i] = true_positives / np.sum(confusion[i, :])

        macro_r = np.sum(r) / N

        return r, macro_r

    @staticmethod
    def f1_score(confusion):
        """ Computes the f1 score per class given a confusion matrix.
        
        Also returns the macro-averaged f1-score across classes.
        
        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.
        
        Returns
        -------
        np.array
            A C-dimensional numpy array, with the f1 score for each
            class in the same order as given in the confusion matrix.
        
        float
            The macro-averaged f1 score across C classes.   
        """

        # Initialise array to store recall for C classes
        f = np.zeros((len(confusion),))

        #######################################################################
        #                 ** YOUR TASK: COMPLETE THIS METHOD **
        #######################################################################

        (p, _) = Evaluator.precision(confusion)
        (r, _) = Evaluator.recall(confusion)
        n = confusion.shape[0]
        for i in range(n):
            cur_p = p[i]
            cur_r = r[i]
            if (cur_p + cur_r) == 0:
                f[i] = 0
            else:
                f[i] = 2 * cur_p * cur_r / (cur_p + cur_r)

        macro_f = np.sum(f) / n

        return f, macro_f

    @staticmethod
    def get_accuracy(predicted, true):
        confusion = Evaluator.confusion_matrix(predicted, true)
        return Evaluator.accuracy(confusion)






