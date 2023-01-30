import torch
import torch.nn as nn


class StyleLoss(nn.Module):
    def gram_matrix(self, features, normalize=True):
        """
        Compute the Gram matrix from features.

        Inputs:
        - features: PyTorch Variable of shape (N, C, H, W) giving features for
          a batch of N images.
        - normalize: optional, whether to normalize the Gram matrix
            If True, divide the Gram matrix by the number of neurons (H * W * C)

        Returns:
        - gram: PyTorch Variable of shape (N, C, C) giving the
          (optionally normalized) Gram matrices for the N input images.
        """
        ##############################################################################
        # TODO: Implement content loss function                                      #
        # Please pay attention to use torch tensor math function to finish it.       #
        # Otherwise, you may run into the issues later that dynamic graph is broken  #
        # and gradient can not be derived.                                           #
        #                                                                            #
        # HINT: you may find torch.bmm() function is handy when it comes to process  #
        # matrix product in a batch. Please check the document about how to use it.  #
        ##############################################################################
        N, C, H, W = features.size()

        # features_flat = torch.flatten(features, start_dim=2, end_dim=3)
        # features_flat_transpose = features_flat.permute((0, 2, 1))
        gram = torch.bmm(torch.flatten(features, start_dim=2, end_dim=3), 
                         torch.flatten(features, start_dim=2, end_dim=3).permute((0, 2, 1)))
        # assert features.requires_grad == True
        if normalize:
            gram = gram / float(N * H * W * C)
        
        # assert gram.requires_grad == True
        return gram
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

    def forward(self, feats, style_layers, style_targets, style_weights):
        """
        Computes the style loss at a set of layers.

        Inputs:
        - feats: list of the features at every layer of the current image, as produced by
          the extract_features function.
        - style_layers: List of layer indices into feats giving the layers to include in the
          style loss.
        - style_targets: List of the same length as style_layers, where style_targets[i] is
          a PyTorch Variable giving the Gram matrix the source style image computed at
          layer style_layers[i].
        - style_weights: List of the same length as style_layers, where style_weights[i]
          is a scalar giving the weight for the style loss at layer style_layers[i].

        Returns:
        - style_loss: A PyTorch Variable holding a scalar giving the style loss.
        """

        ##############################################################################
        # TODO: Implement content loss function                                      #
        # Please pay attention to use torch tensor math function to finish it.       #
        # Otherwise, you may run into the issues later that dynamic graph is broken  #
        # and gradient can not be derived.                                           #
        #                                                                            #
        # Hint:                                                                      #
        # you can do this with one for loop over the style layers, and should not be #
        # very much code (~5 lines). Please refer to the 'style_loss_test' for the   #
        # actual data structure.                                                     #
        #                                                                            #
        # You will need to use your gram_matrix function.                            #
        ##############################################################################
        style_loss = 0
        for i in range(len(style_layers)):
            style_current = feats[style_layers[i]]
            gram_current = self.gram_matrix(style_current)
            gram_target = style_targets[i]
            wt = style_weights[i]
            style_loss += wt * torch.sum(torch.pow(gram_current - gram_target, 2))
        return style_loss
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
