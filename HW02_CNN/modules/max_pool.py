from types import prepare_class
import numpy as np

class MaxPooling:
    '''
    Max Pooling of input
    '''
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        '''
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        '''
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        N, C, H, W  = x.shape
        stride      = self.stride
        kernel_size = self.kernel_size

        row_padding = H % stride + abs(kernel_size - stride)
        col_padding = W % stride + abs(kernel_size - stride)

        row_padding1 = row_padding // 2
        row_padding2 = row_padding // 2 + row_padding % 2

        col_padding1 = col_padding // 2
        col_padding2 = col_padding // 2 + col_padding % 2

        x_pad = np.pad(x, ((0, 0), (0, 0), (row_padding1, row_padding2), (col_padding1, col_padding2)),
                        'constant', constant_values=(0))
        
        H_out = x_pad.shape[2] // 2
        W_out = x_pad.shape[3] // 2

        out = np.empty((N, C, H_out, W_out))
        for n in range(N):
            for c in range(C):
                img = x[n, c, :, :]
                for i in range(H_out):
                    for j in range(W_out):
                        sub_img = x[n, c, i*stride : i*stride + kernel_size,
                                    j*stride : j*stride + kernel_size]
                        out[n][c][i][j] = np.max(sub_img)        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        '''
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        '''
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        out         = self.forward(x)     # after performing forward pass
        stride      = self.stride
        kernel_size = self.kernel_size

        N, C, H, W = x.shape

        row_padding = H % stride + abs(kernel_size - stride)
        col_padding = W % stride + abs(kernel_size - stride)

        row_padding1 = row_padding // 2
        row_padding2 = row_padding // 2 + row_padding % 2

        col_padding1 = col_padding // 2
        col_padding2 = col_padding // 2 + col_padding % 2

        x_pad = np.pad(x, ((0, 0), (0, 0), (row_padding1, row_padding2), (col_padding1, col_padding2)),
                        'constant', constant_values=(0))

        N, C, H, W = x_pad.shape

        indices_input = np.unravel_index(np.arange(H * W), (H, W))
        indices_output = (indices_input[0] // stride, indices_input[1] // stride)

        dx = np.zeros((N, C, H, W))
        for n in range(N):
            for c in range(C):
                for input_row, input_col, output_row, output_col in zip(indices_input[0], indices_input[1], indices_output[0], indices_output[1]):
                    if x_pad[n][c][input_row][input_col] == out[n][c][output_row][output_col]:
                        dx[n][c][input_row][input_col] = dout[n][c][output_row][output_col]
        
        # remove padding
        self.dx = dx[:, :, row_padding1:dx.shape[2] - row_padding2, col_padding1:dx.shape[3]-col_padding2]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
