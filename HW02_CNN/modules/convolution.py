import numpy as np

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        '''
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        '''
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        '''
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        # pad the images
        x = np.array(x)
        x_pad = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=(0))
        in_channels = self.in_channels
        out_channels = self.out_channels
        N, C, H, W = np.shape(x)

        kernel_size = self.kernel_size
        stride = self.stride

        weight_mat = self.weight        
        
        H_new = (int)((H + 2 * self.padding - kernel_size) / stride + 1)
        W_new = (int)((W + 2 * self.padding - kernel_size) / stride + 1)

        out = np.empty((N, out_channels, H_new, W_new))
        for n in range(N):
            sample = x_pad[n, :, :, :]
            for c in range(out_channels):
                # create the filter using weights at this channel
                kernel = weight_mat[c, :, :, :]
                for i in range(H_new):
                    for j in range(W_new):
                        img_sub = sample[:, i*stride : i*stride + kernel_size, j*stride : j*stride + kernel_size]
                        out[n][c][i][j] = np.sum(img_sub * kernel) + self.bias[c]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        '''
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        '''
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        N, C, H, W = x.shape  

        wts_mat = self.weight
        stride  = self.stride
        kernel_size = self.kernel_size

        dx = np.zeros((N, C, H + 2*self.padding, W + 2*self.padding))
        dw = np.zeros([N, *wts_mat.shape])
        db = np.zeros(self.out_channels)

        x_pad = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 
            'constant', constant_values=(0))

        for n in range(N):
            for i in range(dout.shape[2]):
                for j in range(dout.shape[3]):
                    for c in range(self.out_channels):
                        grad_ij     = dout[n, c, i, j]
                        input_ij    = grad_ij * x_pad[n, :, i*stride:i*stride + kernel_size, j*stride:j*stride + kernel_size]
                        wts_ij      = grad_ij * wts_mat[c, :, :, :]
                        dx[n, :, i*stride:i*stride + kernel_size, j*stride:j*stride + kernel_size] += wts_ij
                        dw[n, c, :, :, :] += input_ij
        
        # remove padding
        self.dx = dx[:, :, self.padding:dx.shape[2] - self.padding, self.padding:dx.shape[3]-self.padding]
        self.dw = np.sum(dw, axis=0)
        self.db = np.sum(dout, (0, 2, 3))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################