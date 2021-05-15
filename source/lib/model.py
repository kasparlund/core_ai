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

"""
class Conv2d_cai (torch.nn.Conv2d):
    def __init__ (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', dropout_ratio=-1):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, 
        dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.register_buffer("dropout_ratio",self.dropout_ratio)
        
    def forward(self, x): 
        return super().forward(x)
    #print(dict(learn.model.named_buffers()).keys())
"""

class GeneralRelu(nn.Module):
    def __init__(self, leak=None, sub=None, maxv=None):
        super().__init__()
        self.leak,self.sub,self.maxv = leak,sub,maxv

    def forward(self, x):
        x = F.leaky_relu(x,self.leak, inplace=True) if self.leak is not None else F.relu(x, inplace=True)
        if self.sub is not None: x.sub_(self.sub)
        if self.maxv is not None: x.clamp_max_(self.maxv)
        return x

class ReLUOffset(torch.nn.ReLU):
    def __init__(self, post_relu_offset = 0.15 ): #0.15915):
        super().__init__(inplace=True)
        self.register_buffer("offset",torch.tensor(post_relu_offset, dtype=torch.float32))
    def forward(self, x):
        x = super().forward(x+self.offset) - self.offset
        return x

class Maxout(nn.Module):
    #here is the linear combination with the maxout

    def __init__(self, pool_size, d_out=None):
        super().__init__()
        self.lin, self.pool_size, self.d_out = None, pool_size, d_out

    def forward(self, x):
        self.shape = list(x.size())
        if self.lin is None:
            x        = x.view(self.shape[0],-1)
            n_out    = x.shape[1] if self.d_out is None else self.d_out
            self.lin = nn.Linear(x.shape[1], n_out * self.pool_size) 
            init.kaiming_uniform_(self.lin.weight, a=0, nonlinearity="relu")
            self.lin.bias.data.zero_()
            #self.d_out = self.lin.weight.shape[0] // self.pool_size
            #print(f"shape:{shape} inputs.shape:{x.shape} self.lin.weight.shape:{self.lin.weight.shape}")
        if self.d_out is not None:
            self.shape[-1] = self.d_out


        x   = x.view(self.shape[0],-1)
        #x_shape = x.shape
        x   = self.lin(x)
        #print(f"0 before <=> after nn.linear x.shape:{x_shape} <=> x1.shape:{x1.shape}")
        x    = x.view(x.shape[0], x.shape[1] // self.pool_size, self.pool_size)
        #print(f"1 x.shape:{x.shape} ")
        m, i = x.max(len(x.shape)-1)
        m    = m.view(*self.shape)
        #print(f"m.shape:{m.shape} ")
        #1/0
        return m   

def conv(ni, nf, ks=3, stride=1, bias=False, dropout_ratio=0.25):
    if dropout_ratio > 0. :
        #return [ torch.nn.Dropout(p=dropout_ratio, inplace=False),
        #         torch.nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)]
        return [ torch.nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias),
                 torch.nn.Dropout(p=dropout_ratio, inplace=False) ]
    else:                    
        return [torch.nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)]


def conv_layer(ni, nf, ks, stride, bn, zero_bn, act, dropout_ratio=0.15):
    #ni:      number of input filters
    #nf:      number of output filteres
    #ks:      kernel size
    #act:     activation function : nn.ReLU, nn.reLU
    #bn:      create batchnorm layer
    #zero_bn: init bias and weright in batchnorm to zero
    layers = [*conv(ni, nf, ks, stride=stride, dropout_ratio=dropout_ratio)]
    #"""
    if bn: 
        bnorm = nn.BatchNorm2d(nf)
        nn.init.constant_(bnorm.weight, 0. if zero_bn else 1.)
        layers.append(bnorm)
    #"""
    if act is not None: 
        #layers.append(torch.nn.Dropout(p=dropout_ratio, inplace=False))
        layers.append(act())
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, expansion, ni, nh, stride, activ_func, dpr=0.5):
        super().__init__()

        ni,nf   = ni*expansion, nh*expansion
        layers  = [conv_layer(ni, nh, 1, stride=1,      bn=True, zero_bn=False, act=activ_func, dropout_ratio=dpr)]

        if expansion==1 :
            layers += [ conv_layer(nh, nf, 3, stride=stride, bn=True, zero_bn=True,  act=activ_func, dropout_ratio=dpr) ] 
        else: 
            layers += [ conv_layer(nh, nh, 3, stride=stride, bn=True, zero_bn=False, act=activ_func, dropout_ratio=dpr),
                        conv_layer(nh, nf, 1, stride=1,      bn=True, zero_bn=True,  act=activ_func, dropout_ratio=dpr) ]

        self.convs = nn.Sequential(*layers)
        #print(f"expansion, stride, ni==nf:{expansion}, {stride}, {ni}, {nf}")
        self.pool   = noop if stride==1 else nn.AvgPool2d(2, ceil_mode=True, padding=0 if ni == 2*(ni//2) else 1)
        self.idconv = noop if ni==nf    else conv_layer(ni, nf, 1, stride=1, bn=True, zero_bn=False, act=activ_func, dropout_ratio=dpr)

        #self.dropout =   nn.Dropout(p=dpr)
        #self.bn     = nn.BatchNorm2d(nf)

        #self.bn_org = nn.BatchNorm2d(nf)
        #self.act_fn_org  = activ_func() #nn.LeakyReLU(negative_slope=1e-2) #ReLUOffset()

        self.act_fn  = activ_func() #nn.LeakyReLU(negative_slope=1e-2) #ReLUOffset()
        #nn.init.constant_(self.bn.weight, .5)
        #nn.init.constant_(self.bn.bias, 0.)
        #nn.init.constant_(self.bn_org.weight, .5)
        #nn.init.constant_(self.bn_org.bias, 0.)

    def forward(self, x): 
        #return self.act_fn(self.convs(x).add_( self.idconv(self.pool(x))) )
        #return self.act_fn( self.convs(x) + self.idconv(self.pool(x)) )

        #x = self.convs(x) + self.act_fn_org( self.bn_org(self.idconv(self.pool(self.dropout(x))) ))
        #return self.act_fn( self.bn(x) )

        #return self.act_fn( self.convs(x) + self.idconv(self.pool(self.dropout(x))) ) 

        return self.convs(x) + self.idconv(self.pool(x))
        #return self.act_fn( self.convs(x) + self.idconv(self.pool(x)) )
        #return self.act_fn( self.bn(x) )


class XResNet(nn.Sequential):
    #c_in=1 is configured for mnist
    @classmethod
    def create(cls, expansion, layers, c_in=3, c_out=1000, activ_func = partial(nn.ReLU,inplace=True)):
        dp = 0.15
        #nfs, strides, dpr  = ([c_in, (c_in+1)*8],         [2],    [.0,.15]) if c_in==1 else  \
        #nfs, strides, dpr  = ([c_in, (c_in+1)*8, 32],     [1,2],  [.0,.15,.15]) if c_in==1 else  \
        #nfs, strides, dpr  = ([c_in, (c_in+1)*8, 16], [1,2,1],  [.0,.15,.15]) if c_in==1 else  \
        #nfs, strides, dpr  = ([c_in, (c_in+1)*8, 32, 32], [1,2,1], [dp,dp,dp]) if c_in==1 else  \
        nfs, strides, dpr  = ([c_in, (c_in+1)*8, 32], [1,2], [dp,dp,dp]) if c_in==1 else  \
                             ([c_in, (c_in+1)*8, 32, 64], [1,2,2], [.1,dp,dp,dp])
        stem    = [conv_layer( nfs[i], nfs[i+1], ks=3, stride=strides[i], 
                               act=activ_func, bn=True, zero_bn=False, dropout_ratio=dpr[i]) for i in range(len(nfs)-1)]

        #nfs, strides  = ([16//expansion,16,32,32],       [1,2,1]) if c_in==1 else  \
        #nfs, strides  = ([16//expansion,64,128],      [2,2,1]) if c_in==1 else  \
        #nfs, strides  = ([32//expansion,64,128],      [2,2,1]) if c_in==1 else  \
        #nfs, strides  = ([32//expansion,64,128],         [2,2]) if c_in==1 else  \
        #nfs, strides  = ([32//expansion,64,128,256],     [2,1,2]) if c_in==1 else  \
        nfs, strides  = ([32//expansion,64,256],     [2,2]) if c_in==1 else  \
                        ([64//expansion,64,128,256,512], [1,2,2,2]) 
        if len(nfs) < len(layers)+1: layers=layers[0:len(nfs)-1]
        res_layers = [cls._make_layer(expansion, nfs[i], nfs[i+1], n_blocks=l, stride=strides[i], 
                                      activ_func=activ_func,dpr=dp) for i,l in enumerate(layers) ]

        #n2 =  int(0.5+0.5*(nfs[-1]*expansion+c_out))  
        n2_res = nfs[-1]*expansion                                   
        n2 =  int( 0.5*(nfs[-1]*expansion+c_out))
        res = cls(
            *stem,
            #nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            #nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            *res_layers,
            #nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            #nn.AdaptiveAvgPool2d(1), 
            Flatten(),
            #nn.BatchNorm1d(n2_res),
            nn.Linear(n2_res, c_out)
        )
        """
            nn.Dropout(p=dp, inplace=False),
            nn.Linear(n2_res, n2),
            nn.BatchNorm1d(n2),
            activ_func(),

            nn.BatchNorm1d(n2),
            nn.Dropout(p=dp, inplace=False),
            nn.Linear(n2, c_out)
        """
        return res

    @staticmethod
    def _make_layer(expansion, ni, nf, n_blocks, stride, activ_func,dpr):
        blocks = [ResBlock(expansion, (ni if i==0 else nf), nf, stride=stride, 
                           activ_func=activ_func,dpr=dpr) for i in range(n_blocks)]   
        return nn.Sequential(*blocks)   

    def initialize(self, uniform:bool=False, a=0., nonlinearity="relu"):
        modules = find_submodules(self, lambda m: not isinstance(m, nn.Sequential))
        for m in modules:
            if isinstance(m, (nn.Conv2d,nn.Linear) ):
                init.kaiming_uniform_(m.weight, a=a, nonlinearity=nonlinearity) if uniform else \
                init.kaiming_normal_( m.weight, a=a, nonlinearity=nonlinearity)
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
