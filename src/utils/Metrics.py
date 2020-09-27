import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Function
from pytorch_msssim import SSIM, MS_SSIM



class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


class DiceLoss(torch.nn.Module):

    def init(self):
        super(DiceLoss, self).init()

    def forward(self, input, target):
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = 1 - (2 * self.inter.float() + eps) / self.union.float()
        return t


class DiceBCELoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):     
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE


class JaccardIndex(torch.nn.Module):
    def init(self):
        super(JaccardIndex, self).init()

    def forward(self, input, target):
        eps = 0.0001
        self.inter = torch.sum(torch.dot(input.view(-1), target.view(-1)))
        self.total = torch.sum(input) + torch.sum(target) + eps
        self.union = self.total - self.inter

        t = (self.inter.float() + eps) / self.union.float()
        return t


class JaccardLoss(torch.nn.Module):
    def init(self):
        super(JaccardLoss, self).init()

    def forward(self, input, target):
        eps = 0.0001
        self.inter = torch.sum(torch.dot(input.view(-1), target.view(-1)))
        self.total = torch.sum(input) + torch.sum(target) + eps
        self.union = self.total - self.inter

        t = 1 - (self.inter.float() + eps) / self.union.float()
        return t


class SSIMLoss(torch.nn.Module):
    def init(self):
        super(SSIMLoss, self).init()

    def forward(self, input, target):
        ssim = SSIM(data_range=1, size_average=True, channel=1)
        
        t = 1 - ssim(input, target)
        return t


class MSSSIMLoss(torch.nn.Module):
    def init(self):
        super(SSIMLoss, self).init()

    def forward(self, input, target):
        msssim = MS_SSIM(data_range=1, size_average=True, channel=1)
        
        t = 1 - msssim(input, target)
        return t


class MSESSIMLoss(torch.nn.Module):
    def init(self):
        super(MSESSIMLoss, self).init()

    def forward(self, input, target, alpha=0.35):
        ssim = SSIM(data_range=1, size_average=True, channel=1)
        mse = torch.nn.functional.mse_loss(input, target)
        ssim_loss = 1 - ssim(input, target)

        return (1-alpha)*ssim_loss + alpha*mse


class MAESSIMLoss(torch.nn.Module):
    def init(self):
        super(MSESSIMLoss, self).init()

    def forward(self, input, target, alpha=0.35):
        ssim = SSIM(data_range=1, size_average=True, channel=1)
        mae = torch.nn.functional.l1_loss(input, target)
        ssim_loss = 1 - ssim(input, target)

        return (1-alpha)*ssim_loss + alpha*mae



class MixMSEMAELoss(torch.nn.Module):
    def init(self):
        super(MSESSIMLoss, self).init()

    def forward(self, input, target, alpha=0.35):
        mse = torch.nn.functional.mse_loss(input, target)
        mae = torch.nn.functional.l1_loss(input, target)

        return (1-alpha)*mse + alpha*mae
        

class MixedSSIMLoss(torch.nn.Module):
    def init(self):
        super(MSESSIMLoss, self).init()

    def forward(self, input, target, alpha=0.35):
        ssim = SSIM(data_range=1, size_average=True, channel=1)
        msssim = MS_SSIM(data_range=1, size_average=True, channel=1)

        ssim_loss = 1 - ssim(input, target)
        msssim_loss = 1 - msssim(input, target)
        return (1-alpha)*ssim_loss + alpha*(1-msssim_loss)



def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def jaccard_index(input, target):
    """jaccard index for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + JaccardIndex().forward(c[0], c[1])

    return s / (i + 1)


def jaccard_loss(input, target):
    """jaccard index for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + JaccardLoss().forward(c[0], c[1])

    return s / (i + 1)