import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

def apply_quant(x, nbits):
    x = x.floor()
    if nbits < 32:
        # import ipdb as pdb; pdb.set_trace()
        max_val = 2**(nbits-1)-1
        min_val = -2**(nbits-1)+1
        total_range = max_val - min_val + 1
        mask = x>max_val
        mask = mask.float()
        x    = max_val*mask + (1-mask)*x
        mask = x<min_val
        mask = mask.float()
        x    = min_val*mask + (1-mask)*x
    return x

class IntNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, nbits):
        ctx.save_for_backward(x)
        return apply_quant( x, nbits)
    
    @staticmethod
    def backward(ctx, dx):
        return dx, None

int_nn = IntNN.apply




class BNNSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # import ipdb as pdb; pdb.set_trace()
        ctx.save_for_backward(x)
        return x.sign()
    
    @staticmethod
    def backward(ctx, dx):
        # import ipdb as pdb; pdb.set_trace()
        x, = ctx.saved_variables
        gt1  = x > +1
        lsm1 = x < -1
        gi   = 1-gt1.float()-lsm1.float()
        
        return gi*dx
bnn_sign = BNNSign.apply

