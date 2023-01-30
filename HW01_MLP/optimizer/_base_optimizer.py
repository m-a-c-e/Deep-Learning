class _BaseOptimizer:
    def __init__(self, learning_rate=1e-4, reg=1e-3):
        self.learning_rate = learning_rate
        self.reg = reg

    def update(self, model):
        pass

    def apply_regularization(self, model):
        '''
        Apply L2 penalty to the model. Update the gradient dictionary in the model
        :param model: The model with gradients
        :return: None, but the gradient dictionary of the model should be updated
        '''

        #############################################################################
        # TODO:                                                                     #
        #    1) Apply L2 penalty to model weights based on the regularization       #
        #       coefficient                                                         #
        #############################################################################
        num_layers = int(len(model.weights)) + 1
        if model.weights.get('b1') is not None:
            num_layers = int(len(model.weights) / 2) + 1
        
        for i in range(1, num_layers):
            model.gradients['W' + str(i)] += self.reg * model.weights['W' + str(i)]
        return None
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################