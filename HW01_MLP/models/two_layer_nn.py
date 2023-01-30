# Do not use packages that are not in standard distribution of python
import numpy as np
np.random.seed(1024)
from ._base_network import _baseNetwork

class TwoLayerNet(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10, hidden_size=128):
        super().__init__(input_size, num_classes)

        self.hidden_size = hidden_size
        self._weight_init()


    def _weight_init(self):
        '''
        initialize weights of the network
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the first layer of shape (num_features, hidden_size)
        - b1: The bias term of the first layer of shape (hidden_size,)
        - W2: The weight matrix of the second layer of shape (hidden_size, num_classes)
        - b2: The bias term of the second layer of shape (num_classes,)
        '''

        # initialize weights
        self.weights['b1'] = np.zeros(self.hidden_size)
        self.weights['b2'] = np.zeros(self.num_classes)
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.hidden_size)
        np.random.seed(1024)
        self.weights['W2'] = 0.001 * np.random.randn(self.hidden_size, self.num_classes)

        # initialize gradients to zeros
        self.gradients['W1'] = np.zeros((self.input_size, self.hidden_size))
        self.gradients['b1'] = np.zeros(self.hidden_size)
        self.gradients['W2'] = np.zeros((self.hidden_size, self.num_classes))
        self.gradients['b2'] = np.zeros(self.num_classes)

    def forward(self, X, y, mode='train'):
        '''
        The forward pass of the two-layer net. The activation function used in between the two layers is sigmoid, which
        is to be implemented in self.,sigmoid.
        The method forward should compute the loss of input batch X and gradients of each weights.
        Further, it should also compute the accuracy of given batch. The loss and
        accuracy are returned by the method and gradients are stored in self.gradients

        :param X: a batch of images (N, input_size)
        :param y: labels of images in the batch (N,)
        :param mode: if mode is training, compute and update gradients;else, just return the loss and accuracy
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
            self.gradients: gradients are not explicitly returned but rather updated in the class member self.gradients
        '''
        loss = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process:                                      #
        #        1) Call sigmoid function between the two layers for non-linearity  #
        #        2) The output of the second layer should be passed to softmax      #
        #        function before computing the cross entropy loss                   #
        #    2) Compute Cross-Entropy Loss and batch accuracy based on network      #
        #       outputs                                                             #
        #############################################################################
        X = np.array(X)                     # input matrix
        y = np.array(y)                     # output labels
        num_layers = int(len(self.weights) / 2) + 1
        batch_size = X.shape[0]
        inputs = []
        outputs = [X]
        for i in range(1, num_layers):
            wts_i = self.weights['W' + str(i)]      # get the weights for this layer
            bias_i = self.weights['b' + str(i)]     # get the bias
            in_i = outputs[i - 1] @ wts_i + bias_i           # calculate input
            inputs.append(in_i)                     # append inputs
            if i == 1:
                # sigmoid
                out_i = self.sigmoid(in_i)              # sigmoid
                outputs.append(out_i)                   # append outputs
            else:
                # softmax
                out_i = self.softmax(in_i)
                outputs.append(out_i)
        # outputs[-1] = self.softmax(np.copy(outputs[-1]))     # softmax for the last layer
        loss     = self.cross_entropy_loss(np.copy(outputs[-1]), y)
        accuracy = self.compute_accuracy(np.copy(outputs[-1]), y)    
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
        #    HINT: You will need to compute gradients backwards, i.e, compute       #
        #          gradients of W2 and b2 first, then compute it for W1 and b1      #
        #          You may also want to implement the analytical derivative of      #
        #          the sigmoid function in self.sigmoid_dev first                   #
        #############################################################################
        for i in reversed(range(1, num_layers)):
            # delta_error (softmax + cross-entropy) -> (pi - yi)
            y_one_hot = np.zeros((batch_size, outputs[i].shape[1]))
            y_one_hot[range(batch_size), y] = 1

            if i == num_layers - 1:
                delta_error = outputs[i] - y_one_hot
                self.gradients['W' + str(i)] = delta_error
                self.gradients['b' + str(i)] = delta_error
            else:
                wts_i1 = self.weights['W' + str(i + 1)]
                grad_i1 = self.gradients['W' + str(i + 1)]
                grad_i = self.sigmoid_dev(inputs[i - 1])
                self.gradients['W' + str(i)] = grad_i * (grad_i1 @ wts_i1.T)
                self.gradients['b' + str(i)] = grad_i * (grad_i1 @ wts_i1.T)
        
        for i in reversed(range(1, num_layers)):
            self.gradients['W' + str(i)] = np.copy(outputs[i - 1].T @ self.gradients['W' + str(i)] / batch_size)
            self.gradients['b' + str(i)] = np.copy(np.expand_dims(np.mean(self.gradients['b' + str(i)], axis=0), axis=0))
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss, accuracy


