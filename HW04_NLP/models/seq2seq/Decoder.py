import random

import torch
import torch.nn as nn
import torch.optim as optim

import pdb

class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model
        You will need to complete the init function and the forward function.
    """

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2, model_type="RNN"):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the decoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN", "LSTM".                                                   #
        #       3) A single linear layer with a (log)softmax layer for output       #
        #       4) A dropout layer                                                  #
        #############################################################################
        # Embedding layer
        self.emb_layer = nn.Embedding(self.output_size, self.emb_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Only one recurrent block
        self.rnn_layer = None
        if self.model_type is "RNN":
            self.rnn_layer = nn.RNN(self.emb_size, self.decoder_hidden_size, 1, batch_first=True)
        else:
            self.rnn_layer = nn.LSTM(self.emb_size, self.decoder_hidden_size, 1)

        # Additional layers after RNN output
        self.l1   = nn.Linear(self.decoder_hidden_size, self.output_size)
       

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input, hidden):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, seq_len); HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden weights of the previous time step from the decoder
            Returns:
                output (tensor): the output of the decoder
                hidden (tensor): the weights coming out of the hidden unit
        """
        output = None
        #############################################################################
        # TODO: Implement the forward pass of the encoder.                          #
        #      Please apply the dropout to the embedding layer                      #
        #############################################################################
        batch, seq_len = input.size()
        seq_feats = self.emb_layer(input)
        seq_feats = self.dropout(seq_feats)
        output, hidden = self.rnn_layer(seq_feats, hidden)
        output = self.l1(output)
        output = torch.log_softmax(output, dim=2)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return output[:, 0, :], hidden

