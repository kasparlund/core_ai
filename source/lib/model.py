from __future__ import annotations
from torch import *
from torch import nn
from torch import tensor
from torch.nn import init
import torch.nn.functional as F
import torch as torch
from functools import partial
import math
from lib.callbacks import *

def annealer(f):
    def _inner(start, end): return partial(f, start, end)
    return _inner

@annealer
def sched_lin(start, end, pos): return start + pos*(end-start)

@annealer
def sched_cos(start, end, pos): return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2
@annealer
def sched_no(start, end, pos):  return start
@annealer
def sched_exp(start, end, pos): return start * (end/start) ** pos

def combine_scheds(pcts, scheds):
    assert sum(pcts) == 1.
    pcts = tensor([0] + pcts)
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)
    def _inner(pos):
        idx = (pos >= pcts).nonzero().max()
        actual_pos = (pos-pcts[idx]) / (pcts[idx+1]-pcts[idx])
        return scheds[idx](actual_pos)
    return _inner


def normalize(x, m, s): return (x-m)/s

def normalize_to(train, valid):
    m,s = train.mean(),train.std()
    return normalize(train, m, s), normalize(valid, m, s)

#model functionality
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, x): return self.func(x)
    
def flatten(x):      return x.view(x.shape[0], -1)

def view_tfm(*size):
    def _inner(x): return x.view(*((-1,)+size))
    return _inner

def children(m  ): return list(m.children())

########################## Generalized ReLU ########################
def init_cnn_(m, f):
    if isinstance(m, nn.Conv2d):
        f(m.weight, a=0.1)
        if getattr(m, 'bias', None) is not None: m.bias.data.zero_()
    for l in m.children(): init_cnn_(l, f)

def init_cnn(m, uniform=False):
    f = init.kaiming_uniform_ if uniform else init.kaiming_normal_
    init_cnn_(m, f)

def get_cnn_layers(n_filters_pr_layer,  input_features, output_features, layer, **kwargs):
    nfs = [input_features] + n_filters_pr_layer
    return [layer(nfs[i], nfs[i+1], 5 if i==0 else 3, **kwargs)
            for i in range(len(nfs)-1)] + [
        nn.AdaptiveAvgPool2d(1), Lambda(flatten), nn.Linear(nfs[-1], output_features)]

def get_cnn_model(filters_pr_layer,  input_features,  output_features, layer, **kwargs):
    return nn.Sequential(*get_cnn_layers(filters_pr_layer,  input_features, output_features, layer, **kwargs))



def noop(x): 
    #print(f"noop:{x.shape}")
    return x

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

act_fn = nn.ReLU(inplace=True)


def conv_layer(ni, nf, ks=3, stride=1, zero_bn=False, act=True, bn=True):
    #ni: number of input filters
    #nf: number of output filteres
    #ks: kernel size
    layers = [conv(ni, nf, ks, stride=stride)]
    if bn: 
        bnorm = nn.BatchNorm2d(nf)
        nn.init.constant_(bnorm.weight, 0. if zero_bn else 1.)
        layers.append(bnorm)
    if act: layers.append(act_fn)
    return nn.Sequential(*layers)
"""
def conv_layer(ni, nf, ks=3, stride=2, bn=True, **kwargs):
    layers = [nn.Conv2d(ni, nf, ks, padding=ks//2, stride=stride, bias=not bn),
              GeneralRelu(**kwargs)]
    if bn: layers.append(nn.BatchNorm2d(nf, eps=1e-5, momentum=0.1))
    return nn.Sequential(*layers)
"""

class ResBlock(nn.Module):
    def __init__(self, expansion, ni, nh, stride=1):
        super().__init__()
        nf,ni   = nh*expansion,ni*expansion
        layers  = [conv_layer(ni, nh, 1)]
        layers += [
            conv_layer(nh, nf, 3, stride=stride, zero_bn=True, act=False)
        ] if expansion==1 else [
            conv_layer(nh, nh, 3, stride=stride),
            conv_layer(nh, nf, 1, zero_bn=True, act=False)
        ]
        self.convs = nn.Sequential(*layers)
        #print(f"expansion, stride, ni==nf:{expansion}, {stride}, {ni}, {nf}")
        self.idconv = noop if ni==nf    else conv_layer(ni, nf, 1, act=False)
        self.pool   = noop if stride==1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x): 
        """
        a = self.convs(x)
        b = self.idconv(self.pool(x))
        print(f"\nx.shape:{x.shape}, self.pool(x).shape:{self.pool(x).shape}\na.shape:{a.shape}, b.shape:{b.shape}")
        c = a + b
        #print(f"c.shape: {c.shape}")
        
        return act_fn(c)
        """
        return act_fn(self.convs(x).add_( self.idconv(self.pool(x))) )
def init_cnn_resnet(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn_resnet(l)
class XResNet(nn.Sequential):
    @classmethod
    def create(cls, expansion, layers, c_in=3, c_out=1000):
        nfs  = [c_in, (c_in+1)*8, 64, 64]
        stem = [conv_layer(nfs[i], nfs[i+1], stride=2 if i==0 else 1)
                for i in range(3)]

        nfs  = [64//expansion,64,128,256,512]
        res_layers = [cls._make_layer(expansion, nfs[i], nfs[i+1], n_blocks=l, stride=1 if i==0 else 2)
                      for i,l in enumerate(layers)]
        res = cls(
            *stem,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *res_layers,
            nn.AdaptiveAvgPool2d(1), Flatten(),
            nn.Linear(nfs[-1]*expansion, c_out),
        )
        init_cnn_resnet(res)
        return res

    @staticmethod
    def _make_layer(expansion, ni, nf, n_blocks, stride):
        return nn.Sequential(
            *[ResBlock(expansion, ni if i==0 else nf, nf, stride if i==0 else 1)
              for i in range(n_blocks)])   
    
def xresnet18 (**kwargs): return XResNet.create(1, [2, 2, 2, 2], **kwargs)
def xresnet34 (**kwargs): return XResNet.create(1, [3, 4, 6, 3], **kwargs)
def xresnet50 (**kwargs): return XResNet.create(4, [3, 4, 6, 3], **kwargs)
def xresnet101(**kwargs): return XResNet.create(4, [3, 4, 23, 3], **kwargs)
def xresnet152(**kwargs): return XResNet.create(4, [3, 8, 36, 3], **kwargs)   

def find_modules(m, cond):
    if cond(m): return [m]
    return sum([find_modules(o,cond) for o in m.children()], [])
class GetOneBatchCallback(Callback):
    def after_preprocessing(self, e:Event): 
        self.xb,self.yb = e.learn.xb,e.learn.yb
        e.learn.stop = True
def getFirstbatch(model, data:DataBunch, cbs_tranform:BatchTransformXCallback ):
    cbfs  = [cbs_tranform,GetOneBatchCallback]
    learn = Learner( model, data, loss_func=None, opt=None,cb_funcs=cbfs)
    learn.fit(1)
    cb = learn.find_subcription_by_cls(GetOneBatchCallback)
    return cb.xb, cb.yb
def model_summary(model, xb:Tensor, find_all=False, print_mod=False):
    f = lambda hook,mod,inp,out: print(f"====\n{mod}\n" if print_mod else "", out.shape)
    with Hooks(model, f) as hooks: model(xb)
    
