import torch
import torch.nn as nn
import pdb

class ContentLoss(nn.Module):
    def forward(self, content_weight, content_current, content_original):
        """
        Compute the content loss for style transfer.

        Inputs:
        - content_weight: Scalar giving the weighting for the content loss.
        - content_current: features of the current image; this is a PyTorch Tensor of shape
          (1, C_l, H_l, W_l).
        - content_target: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

        Returns:
        - scalar content loss
        """

        ##############################################################################
        # TODO: Implement content loss function                                      #
        # Please pay attention to use torch tensor math function to finish it.       #
        # Otherwise, you may run into the issues later that dynamic graph is broken  #
        # and gradient can not be derived.                                           #
        ##############################################################################
        # content_current_flat = torch.flatten(content_current, start_dim=2, end_dim=3)
        # content_original_flat = torch.flatten(content_original, start_dim=2, end_dim=3)
        # pdb.set_trace()
        content_loss = content_weight * torch.sum(torch.pow(content_current - content_original, 2))
        return content_loss
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
