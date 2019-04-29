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

##################################################
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
    return [layer(nfs[i], nfs[i+1], ks=(5 if i==0 else 3), **kwargs) for i in range(len(nfs)-1)] + \
           [nn.AdaptiveAvgPool2d(1), Lambda(flatten), nn.Linear(nfs[-1], output_features)]

def get_cnn_model(filters_pr_layer,  input_features,  output_features, layer, **kwargs):
    return nn.Sequential(*get_cnn_layers(filters_pr_layer,  input_features, output_features, layer, **kwargs))
"""
import math
def prev_pow_2(x): return 2**math.floor(math.log2(x))
def get_cnn_layers(data, nfs, layer, **kwargs):
    def f(ni, nf, stride=2): return layer(ni, nf, 3, stride=stride, **kwargs)
    l1 = data.c_in
    l2 = prev_pow_2(l1*3*3)
    layers =  [f(l1  , l2  , stride=1),
               f(l2  , l2*2, stride=2),
               f(l2*2, l2*4, stride=2)]
    nfs = [l2*4] + nfs
    layers += [f(nfs[i], nfs[i+1]) for i in range(len(nfs)-1)]
    layers += [nn.AdaptiveAvgPool2d(1), Lambda(flatten), 
               nn.Linear(nfs[-1], data.c_out)]
    return layers
"""


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
def conv_layer(ni, nf, ks, stride, bn, zero_bn, act):
    #ni:      number of input filters
    #nf:      number of output filteres
    #ks:      kernel size
    #act:     activation function : nn.ReLU, nn.reLU
    #bn:      create batchnorm layer
    #zero_bn: init bias and weright in batchnorm to zero
    layers = [conv(ni, nf, ks, stride=stride)]
    if bn: 
        bnorm = nn.BatchNorm2d(nf)
        nn.init.constant_(bnorm.weight, 0. if zero_bn else 1.)
        layers.append(bnorm)
    if act is not None: 
        layers.append(act())
    return nn.Sequential(*layers)

def init_cnn_resnet(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn_resnet(l)

act_fn = partial(nn.ReLU,inplace=True)()
class ResBlock(nn.Module):
    def __init__(self, expansion, ni, nh, stride, activ_func):
        super().__init__()
        ni,nf   = ni*expansion, nh*expansion
        layers  = [conv_layer(ni, nh, 1, stride=1,      bn=True, zero_bn=False, act=activ_func)]
        layers += [
                   conv_layer(nh, nf, 3, stride=stride, bn=True, zero_bn=True,  act=None)
        ] if expansion==1 else [
                   conv_layer(nh, nh, 3, stride=stride, bn=True, zero_bn=False, act=activ_func),
                   conv_layer(nh, nf, 1, stride=1,      bn=True, zero_bn=True,  act=None)
        ]
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
    def create(cls, expansion, layers, c_in=3, c_out=1000):
        activ_func = partial(nn.ReLU,inplace=True)
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
        init_cnn_resnet(res)
        return res

    @staticmethod
    def _make_layer(expansion, ni, nf, n_blocks, stride, activ_func):
        blocks = [ResBlock(expansion, (ni if i==0 else nf), nf, stride=stride, 
                           activ_func=activ_func) for i in range(n_blocks)]   
        return nn.Sequential(*blocks)   
    
def xresnet18 (**kwargs): return XResNet.create(1, [2, 2, 2,  2], **kwargs)
def xresnet34 (**kwargs): return XResNet.create(1, [3, 4, 6,  3], **kwargs)
def xresnet50 (**kwargs): return XResNet.create(4, [3, 4, 6,  3], **kwargs)
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
    learn = Learner( model, data, loss_func=None)
    learn.fit(1, opt=None,cb_funcs=cbfs)
    cb = learn.find_subcription_by_cls(GetOneBatchCallback)
    return cb.xb, cb.yb

def is_lin_layer(l):
    lin_layers = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
    return isinstance(l, lin_layers)

def model_summary(model, xb:Tensor, find_all=False, print_mod=False):
    print("model_summary")
    #device = next(model.parameters()).device
    #xb     = xb.to(device)
    f      = lambda hook,mod,inp,out: print(f"\n{mod}\n{out.shape}, requires_grad:{out.requires_grad}") if print_mod \
                                      else print(f"{out.shape}, requires_grad={out.requires_grad}")
    mods = find_modules(model, is_lin_layer) if find_all else model.children()
    with Hooks(mods, f) as hooks: model(xb)


################### transfer learning #####################
#save af model
def save_model(path, model):
    mdl_path = path/'models'
    mdl_path.mkdir(exist_ok=True)
    st = model.state_dict()
    torch.save(st, mdl_path/'iw5')
    
def load_model(path, model):
    mdl_path = path/'models'
    st = torch.load(mdl_path/'iw5')    
    model.load_state_dict(st)

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=1):
        super().__init__()
        self.output_size = sz
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

    
    
# batchnorm must not be modified during transferlearning
def freeze( model ):
    #the pretrained part i located i layer 0
    #model.apply( partial(set_grad, require_grad=False) )
    #model[0].apply( partial(set_grad, require_grad=False) )
    for p in model[0].parameters(): p.requires_grad_(False)
    
def unfreeze( model ):
    #the pretrained part i located i layer 0
    model[0].apply( partial(set_grad, require_grad=True) )
    #for p in model[0].parameters(): set_grad(p.requires_grad_(True)
    

"""    
def bn_splitter(m):
    def _bn_splitter(l, g1, g2):
        if isinstance(l, nn.BatchNorm2d): g2 += l.parameters()
        elif hasattr(l, 'weight'): g1 += l.parameters()
        for ll in l.children(): _bn_splitter(ll, g1, g2)
        
    g1,g2 = [],[]
    _bn_splitter(m[0], g1, g2)
    
    g2 += m[1:].parameters()
    return g1,g2
a,b = bn_splitter(learn.model)
"""        