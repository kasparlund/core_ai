from __future__ import annotations
from torch import *
from torch import nn
from torch import tensor
from torch.nn import init
import torch.nn.functional as F
from functools import partial

from .callbacks import *

def find_submodules(module:nn.Module, condition):
    def find(module, condition):
        if condition(module): return [module] 
        else:                 return sum([find(o,condition) for o in module.children()], [])
    return find(module,condition)

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, x): return self.func(x)
    
def flatten(x):      return x.view(x.shape[0], -1)

def get_cnn_layers(n_filters_pr_layer,  input_features, output_features, layer, **kwargs):
    nfs = [input_features] + n_filters_pr_layer
    print(f"channels pr layers from input to output: {nfs+[output_features]}")

    in2hidden_layers     = [layer(nfs[i], nfs[i+1], ks=(5 if i==0 else 3), **kwargs) for i in range(len(nfs)-1)] 
    print(f"number of input and hidden layers: {len(in2hidden_layers)}")

    hidden2output_layers = [nn.AdaptiveAvgPool2d(1), Lambda(flatten), nn.Linear(nfs[-1], output_features)]
    print(f"number of output layers :          {len(hidden2output_layers)}")

    all_layers = in2hidden_layers + hidden2output_layers
    print(f"total number of layers:            {len(all_layers)}")
    return all_layers
    #return [layer(nfs[i], nfs[i+1], ks=(5 if i==0 else 3), **kwargs) for i in range(len(nfs)-1)] + \
    #       [nn.AdaptiveAvgPool2d(1), Lambda(flatten), nn.Linear(nfs[-1], output_features)]

def get_cnn_model(filters_pr_layer,  input_features,  output_features, layer, **kwargs):
    return nn.Sequential(*get_cnn_layers(filters_pr_layer,  input_features, output_features, layer, **kwargs))

def noop(x): return x

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

def conv(ni, nf, ks=3, stride=1, bias=False):
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)

class GeneralRelu(nn.Module):
    def __init__(self, leak=None, sub=None, maxv=None):
        super().__init__()
        self.leak,self.sub,self.maxv = leak,sub,maxv

    def forward(self, x):
        x = F.leaky_relu(x,self.leak, inplace=True) if self.leak is not None else F.relu(x, inplace=True)
        if self.sub is not None: x.sub_(self.sub)
        if self.maxv is not None: x.clamp_max_(self.maxv)
        return x

"""
class ReLUOffset(torch.nn.ReLU):
    def __init__(self, post_relu_offset = -0.15):
        super().__init__(inplace=True)
        self.register_buffer("offset",torch.tensor(post_relu_offset, dtype=torch.float32))
        #self.register_buffer("ix",None)
        #self.register_buffer("offset1",torch.tensor(-0.3, dtype=torch.float32))

    def forward(self, x):
        #with torch.no_grad():
        #ix = x < 0 

        ix0 = x < 0 
        #ix1 = ix0.clone()
        #ix1[ np.random.choice([True,False], size=len(x)) ] = False    
        #ix0[ix1] = False    
    
        x = super().forward(x)

        #x[ix0] = self.offset
        #x[ix1] = self.offset1

        #x = x.masked_fill(ix0, self.offset)
        x = x + torch.normal(x+-0.2,0.05)
        #x = x + x.masked_scatter(ix0,torch.normal(x+-0.15,0.15))

        #x = x.masked_fill(ix1, self.offset1)

        return x
"""
class ReLUOffset(torch.nn.ReLU):
    def __init__(self, post_relu_offset = -0.15):
        super().__init__(inplace=True)
        self.register_buffer("offset",torch.tensor(post_relu_offset, dtype=torch.float32))
    def forward(self, x):
        x = super().forward(x) +self.offset
        return x

act_fn = nn.ReLU(inplace=True)
def conv_layer(ni, nf, ks, stride, bn, zero_bn, act):
    #ni:      number of input filters
    #nf:      number of output filteres
    #ks:      kernel size
    #act:     activation function : nn.ReLU, nn.reLU
    #bn:      create batchnorm layer
    #zero_bn: init bias and weright in batchnorm to zero
    layers = [conv(ni, nf, ks, stride=stride)]
    #"""
    if bn: 
        bnorm = nn.BatchNorm2d(nf)
        nn.init.constant_(bnorm.weight, 0. if zero_bn else 1.)
        layers.append(bnorm)
    #"""
    if act is not None: 
        layers.append(act())
    return nn.Sequential(*layers)

act_fn = partial(nn.ReLU,inplace=True)()
class ResBlock(nn.Module):
    def __init__(self, expansion, ni, nh, stride, activ_func):
        super().__init__()
        ni,nf   = ni*expansion, nh*expansion
        layers  = [conv_layer(ni, nh, 1, stride=1,      bn=True, zero_bn=False, act=activ_func)]

        if expansion==1 :
            layers += [ conv_layer(nh, nf, 3, stride=stride, bn=True, zero_bn=True,  act=None) ] 
        else: 
            layers += [ conv_layer(nh, nh, 3, stride=stride, bn=True, zero_bn=False, act=activ_func),
                        conv_layer(nh, nf, 1, stride=1,      bn=True, zero_bn=True,  act=None) ]

        self.convs = nn.Sequential(*layers)
        #print(f"expansion, stride, ni==nf:{expansion}, {stride}, {ni}, {nf}")
        self.idconv = noop if ni==nf    else conv_layer(ni, nf, 1, stride=1, bn=True, zero_bn=False, act=None)
        self.pool   = noop if stride==1 else nn.AvgPool2d(2, ceil_mode=True, padding=0 if ni == 2*(ni//2) else 1)

    def forward(self, x): 
        #return self.act_fn(self.convs(x).add_( self.idconv(self.pool(x))) )
        return act_fn( self.convs(x) + self.idconv(self.pool(x)) ) 

class XResNet(nn.Sequential):
    #c_in=1 is configured for mnist
    @classmethod
    def create(cls, expansion, layers, c_in=3, c_out=1000, activ_func = partial(nn.ReLU,inplace=True)):
        
        nfs, strides  = ([c_in, (c_in+1)*8],         [2]) if c_in==1 else  \
                        ([c_in, (c_in+1)*8, 64, 64], [1,2,2])
        stem    = [conv_layer( nfs[i], nfs[i+1], ks=3, stride=strides[i], 
                               act=activ_func, bn=True, zero_bn=False) for i in range(len(nfs)-1)]

        nfs, strides  = ([16//expansion,16,32,32],       [1,2,1]) if c_in==1 else  \
                        ([64//expansion,64,128,256,512], [1,2,2,2]) 
        if len(nfs) < len(layers)+1: layers=layers[0:len(nfs)-1]
        res_layers = [cls._make_layer(expansion, nfs[i], nfs[i+1], n_blocks=l, stride=strides[i], 
                                      activ_func=activ_func) for i,l in enumerate(layers) ]
        res = cls(
            *stem,
            #nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            *res_layers,
            nn.AdaptiveAvgPool2d(1), Flatten(),
            #( ( nn.AdaptiveAvgPool2d(1), Flatten() ) if c_in > 1 else Flatten() ),
            nn.Linear(nfs[-1]*expansion, c_out),
        )
        return res

    @staticmethod
    def _make_layer(expansion, ni, nf, n_blocks, stride, activ_func):
        blocks = [ResBlock(expansion, (ni if i==0 else nf), nf, stride=stride, 
                           activ_func=activ_func) for i in range(n_blocks)]   
        return nn.Sequential(*blocks)   

    def initialize(self, uniform:bool=False, a=0.1):
        modules = find_submodules(self, lambda m: not isinstance(m, nn.Sequential))
        for m in modules:
            if isinstance(m, (nn.Conv2d,nn.Linear) ):
                init.kaiming_uniform_(m.weight, a=a) if uniform else init.kaiming_normal_(m.weight, a=a)
                if getattr(m, 'bias', None) is not None: m.bias.data.zero_()
            #elif: isinstance(m, nn.BatchNorm2d):
            #    nn.init.constant_(bnorm.weight, 0. if zero_bn else 1.)

            if getattr(m, 'bias', None) is not None: m.bias.data.zero_()

def xresnet18 (**kwargs): return XResNet.create(1, [2, 2, 2,  2], **kwargs)
def xresnet34 (**kwargs): return XResNet.create(1, [3, 4, 6,  3], **kwargs)
def xresnet50 (**kwargs): return XResNet.create(4, [3, 4, 6,  3], **kwargs)
def xresnet101(**kwargs): return XResNet.create(4, [3, 4, 23, 3], **kwargs)
def xresnet152(**kwargs): return XResNet.create(4, [3, 8, 36, 3], **kwargs)   

################### top is refactored from the code below this line #####################
