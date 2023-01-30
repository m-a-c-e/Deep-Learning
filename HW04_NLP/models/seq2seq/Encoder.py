import random

import torch
import torch.nn as nn
import torch.optim as optim

import pdb

class Encoder(nn.Module):
    """ The Encoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """
    def __init__(self, input_size, emb_size, encoder_hidden_size, decoder_hidden_size, dropout = 0.2, model_type = "RNN"):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.model_type = model_type
        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the encoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN" and "LSTM".                                                #
        #       3) Linear layers with ReLU activation in between to get the         #
        #          hidden weights of the Encoder(namely, Linear - ReLU - Linear).   #
        #          The size of the output of the first linear layer is the same as  #
        #          its input size.                                                  #
        #          HINT: the size of the output of the second linear layer must     #
        #          satisfy certain constraint relevant to the decoder.              #
        #       4) A dropout layer
        #     Note: Use if (RNN) and elif (LSTM) for model_type during initialization #
        #############################################################################
        # Embedding layer
        self.emb_layer = nn.Embedding(self.input_size, self.emb_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Only one recurrent block
        self.rnn_layer = None
        if self.model_type is "RNN":
            self.rnn_layer = nn.RNN(self.emb_size, self.encoder_hidden_size, 1, batch_first=True)
        else:
            # TODO
            self.rnn_layer = nn.LSTM(self.emb_size, self.encoder_hidden_size, 1)

        # Additional layers after RNN output
        self.l1   = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size)
        self.l2   = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input):
        """ The forward pass of the encoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, seq_len)

            Returns:
                output (tensor): the output of the Encoder.
                hidden (tensor): the hidden weights coming out of the last hidden unit; later fed into the Decoder.
        """

        #############################################################################
        # TODO: Implement the forward pass of the encoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #       Apply tanh activation to the hidden tensor before returning it      #
        #############################################################################
        output, hidden = None, None
        
        # 1. Pass the input indices/tokens to embedding layer to generate feature rep
        seq_feats = self.emb_layer(input)
        seq_feats = self.dropout(seq_feats) # apply dropo  

        # 2. Pass the feats to RNN
        output, hidden = self.rnn_layer(seq_feats)

        # 3. Apply additional learnable params
        hidden = self.l1(hidden)
        hidden = torch.relu(hidden)
        hidden = self.l2(hidden)
        hidden = torch.tanh(hidden)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return output, hidden