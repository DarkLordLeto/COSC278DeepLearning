import torch


class MeanSquaredError(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2):
        """
        computes the mean squared error between x1 (inputs) and x2 (targets)

        Arguments
        -------
        ctx: a pytorch context object
        x1: (Tensor of size (T x n) where T is the batch size and n is the number of input features.
        x2: (Tensor) of size (T x n)

        Returns
        ------
        y: (scalar) The mean squared error between x1 and x2
        """
        ctx.save_for_backward(x1, x2)
        y = torch.mean((x1 - x2) ** 2)

        return y

    @staticmethod
    def backward(ctx, dzdy):
        """
        back-propagates the error with respect to the input arguments

        Arguments
        --------
        ctx: A PyTorch context object
        dzdy:  a scalar (Tensor), the gradient with respect to y

        Returns
        ------
        dzdx1 (Tensor): of size(T x n), the gradients w.r.t x1
        dzdx2 (Tensor): of size(T x n), the gradients w.r.t x2
        """

        x1, x2 = ctx.saved_tensors
        t, n = x1.shape
       
        dzdx1 = 2 * dzdy * (x1 - x2) / (t * n)
        dzdx2 = -2 * dzdy * (x1 - x2) / (t * n)

        return dzdx1, dzdx2
