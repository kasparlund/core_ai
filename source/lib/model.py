from __future__ import annotations
from torch import *
from torch import nn
from torch import tensor
from torch.nn import init
import torch as torch
from functools import partial
import math

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

def conv2d(ni, nf, ks=3, stride=2):
    "creates a layer with nn.Conv2d followed by af nn.ReLU"
    #ni: number of input filters
    #nf: number of output filteres
    #ks: kernel size
    return nn.Sequential( nn.Conv2d(ni, nf, ks, padding=ks//2, stride=stride), nn.ReLU() )

def get_cnn_layers_ch1(n_filters_pr_layer, n_out_features):
    #add the input layers with 1 filter pr pixel 
    n_filters_pr_layer = [1] + n_filters_pr_layer
    layers = [ conv2d(n_filters_pr_layer[i], n_filters_pr_layer[i+1], 5 if i==0 else 3)
               for i in range(len(n_filters_pr_layer)-1) ]
    layers.extend( [nn.AdaptiveAvgPool2d(1), Lambda(flatten), nn.Linear(n_filters_pr_layer[-1], n_out_features)] )
    return layers

def get_cnn_model_ch1(n_filters_pr_layer, n_out_features): 
    return nn.Sequential( *get_cnn_layers_ch1(n_filters_pr_layer, n_out_features ) )

def init_cnn_(m, f):
    if isinstance(m, nn.Conv2d):
        f(m.weight, a=0.1)
        if getattr(m, 'bias', None) is not None: m.bias.data.zero_()
    for l in m.children(): init_cnn_(l, f)

def init_cnn(m, uniform=False):
    f = init.kaiming_uniform_ if uniform else init.kaiming_normal_
    init_cnn_(m, f)
    return m