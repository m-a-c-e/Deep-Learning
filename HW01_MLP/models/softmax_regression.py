# Do not use packages that are not in standard distribution of python
import numpy as np

from ._base_network import _baseNetwork

class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28*28, num_classes=10):
        '''
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => ReLU activation => Softmax
        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        '''
        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self):
        '''
        initialize weights of the single layer regression network. No bias term included.
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the linear layer of shape (num_features, hidden_size)
        '''
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

    def forward(self, X, y, mode='train'):
        '''
        Compute loss and gradients using softmax with vectorization.

        :param X: a batch of image (N, 28x28)
        :param y: labels of images in the batch (N,)
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        '''
        loss = None
        gradient = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process and compute the Cross-Entropy loss    #
        # Hint:                                                                     #
        #   Store your intermediate outputs before ReLU for backwards               #
        #############################################################################
        X = np.array(X)     # input matrix
        y = np.array(y)     # output labels
        num_layers = len(self.weights) + 1
        batch_size = X.shape[0]
        inputs = []
        outputs = [X]
        for i in range(1, num_layers):
            wts_i = self.weights['W' + str(i)]      # get the weights for this layer
            in_i = outputs[i - 1] @ wts_i           # calculate input
            inputs.append(in_i)                     # append inputs
            out_i = self.ReLU(in_i)                 # ReLU
            outputs.append(out_i)
        outputs[-1] = self.softmax(outputs[-1])     # softmax for the last layer
        
        loss     = self.cross_entropy_loss(outputs[-1], y)
        accuracy = self.compute_accuracy(outputs[-1], y)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        if mode != 'train':
            return loss, accuracy

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight and bias by chain rule         #
        #        2) Store the gradients in self.gradients                           #
        #############################################################################
        for i in reversed(range(1, num_layers)):
            # delta_error (softmax + cross-entropy) -> (pi - yi)
            y_one_hot = np.zeros((batch_size, outputs[i].shape[1]))
            y_one_hot[range(batch_size), y] = 1
            delta_error = outputs[i] - y_one_hot

            # ReLU grad
            delta_relu = self.ReLU_dev(inputs[i - 1])

            # delta total for this layer
            delta_total = (outputs[i - 1].T @ (delta_relu * delta_error))

            # average out over the batch size
            self.gradients['W' + str(i)] = delta_total / batch_size
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss, accuracy





        


