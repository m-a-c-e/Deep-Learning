# Do not use packages that are not in standard distribution of python
import numpy as np
class _baseNetwork:
    def __init__(self, input_size=28 * 28, num_classes=10):

        self.input_size = input_size
        self.num_classes = num_classes

        self.weights = dict()
        self.gradients = dict()

    def _weight_init(self):
        pass

    def forward(self):
        pass

    def softmax(self, scores):
        '''
        Compute softmax scores given the raw output from the model

        :param scores: raw scores from the model (N, num_classes)
        :return:
            prob: softmax probabilities (N, num_classes)
        '''
        prob = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Calculate softmax scores of input images                            #
        #############################################################################
        scores = np.array(scores)
        scores_exp = np.exp(scores - np.expand_dims(np.max(scores, axis=1), axis=1))
        prob = scores_exp / np.reshape(np.sum(scores_exp, axis=1), (scores_exp.shape[0], 1))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return prob

    def cross_entropy_loss(self, x_pred, y):
        '''
        Compute Cross-Entropy Loss based on prediction of the network and labels
        :param x_pred: Raw prediction from the two-layer net (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The computed Cross-Entropy Loss
        '''
        loss = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement Cross-Entropy Loss                                        #
        #############################################################################
        x_pred = np.array(x_pred)
        y = np.array(y)
        row_indices = range(x_pred.shape[0])
        loss = np.mean(-np.log(x_pred[row_indices, y]), axis=0)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss

    def compute_accuracy(self, x_pred, y):
        '''
        Compute the accuracy of current batch
        :param x_pred: Raw prediction from the two-layer net (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The accuracy of the batch
        '''
        acc = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the accuracy function                                     #
        #############################################################################
        x_pred = np.array(x_pred)
        y = np.array(y)
        x_labels = np.argmax(x_pred, axis=1)
        acc = np.mean(np.where(x_labels == y, 1, 0))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return acc

    def sigmoid(self, X):
        '''
        Compute the sigmoid activation for the input

        :param X: the input data coming out from one layer of the model (N, num_classes)
        :return:
            out: the value after the sigmoid activation is applied to the input (N, num_classes)
        '''
        out = None
        #############################################################################
        # TODO: Comput the sigmoid activation on the input                          #
        #############################################################################
        X = np.array(X)
        out = (1 + np.exp(-X)) ** -1
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out

    def sigmoid_dev(self, x):
        '''
        The analytical derivative of sigmoid function at x
        :param x: Input data
        :return: The derivative of sigmoid function at x
        '''
        ds = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the derivative of Sigmoid function                        #
        #############################################################################
        x = np.array(x)
        ds = np.exp(-x) * np.power((1 + np.exp(-x)), -2)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return ds

    def ReLU(self, X):
        '''
        Compute the ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, num_classes)
        :return:
            out: the value after the ReLU activation is applied to the input (N, num_classes)
        '''
        out = None

        #############################################################################
        # TODO: Comput the ReLU activation on the input                          #
        #############################################################################
        X = np.array(X)
        out = np.where(X < 0, 0, X)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out

    def ReLU_dev(self,X):
        '''
        Compute the gradient ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, num_classes)
        :return:
            out: gradient of ReLU given input X
        '''
        out = None
        #############################################################################
        # TODO: Comput the gradient of ReLU activation                              #
        #############################################################################
        X = np.array(X)
        out = np.where(X < 0, 0, 1)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out
