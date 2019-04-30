from __future__ import annotations
"""
from torch import *
from torch import nn
from torch import tensor
from torch.nn import init
import torch.nn.functional as F
import torch as torch
from functools import partial
"""
from lib.callbacks import *
from lib.model import *
from pathlib import Path

class ModelManager():
    def __init__(self,model):self.model=model

    #@classmethod
    #def create_from_model(model:nn.Module): 
    @staticmethod
    def find_submodules(module:nn.Module, condition):
        def find(module, condition):
            if condition(module): return [module] 
            else:                 return sum([find(o,condition) for o in module.children()], [])
        return find(module,condition)

    def find_modules(self,condition):
        return ModelManager.find_submodules(self.model, condition)

    def summary(self, xb:Tensor, only_leaves=True, print_mod=False):
        #device = next(model.parameters()).device
        #xb     = xb.to(device)
        f      = lambda hook,mod,inp,out: print(f"\n{mod}\n{out.shape}") if print_mod else print(f"{type(mod)} {out.shape}")
        mods = self.find_modules(lambda m: not isinstance(m, nn.Sequential)) if only_leaves else \
               self.model.children() 
        with Hooks(mods, f) as hooks: self.model(xb)

    def grads_summary(self):
        modules = self.find_modules( condition=lambda m: not isinstance(m, nn.Sequential) )
        for module in modules:
            if len(list(module.children()))==0:
                requires_grad     = [p.requires_grad for p in module.parameters(recurse=False)]
                str_requires_grad = "None "    
                if len(requires_grad) > 0:    
                    str_requires_grad = "False" if sum(requires_grad) == 0 else "True " if sum(requires_grad)==len(requires_grad) else "None"
                print(f"requires_grad: {str_requires_grad} : {type(module).__name__}")

    def save(self, path, subdir="models"):
        mdl_path = Path(path)/subdir
        mdl_path.mkdir(exist_ok=True)
        st = self.model.state_dict()
        torch.save(st, mdl_path/'iw5')
    
    def load(self, path, subdir="models"):
        mdl_path = Path(path)/subdir
        st = torch.load(mdl_path/'iw5')    
        self.model.load_state_dict(st)

    @staticmethod
    def set_grad(module, requires_grad, train_bn=False):
        if isinstance(module, (nn.BatchNorm2d)): return

        for p in module.parameters(recurse=False):
            p.requires_grad_(requires_grad)

    def change_requires_grad_(self, modules:Collection[nn.Module], requires_grad, train_bn):
        condition = lambda m: not isinstance(m, nn.Sequential)
        selection = []
        for m in modules:   selection.extend( ModelManager.find_submodules(m, condition) )
        for m in selection: ModelManager.set_grad(m, requires_grad, train_bn)
        
    def freeze( self, train_bn=False ):
        self.change_requires_grad_([self.model[0]], requires_grad=False, train_bn=train_bn)    
        self.change_requires_grad_(self.model[1:],  requires_grad=True,  train_bn=train_bn)
    
    def unfreeze( self, train_bn=False ):
        self.change_requires_grad_(self.model,    requires_grad=True, train_bn=train_bn)    

    def getFirstbatch(self, databunch:DataBunch, normalization:Callback ):
        cbfs  = [partial(BatchTransformXCallback, tfm = normalization), GetOneBatchCallback]
        learn = Learner( self.model, databunch, loss_func=None)
        learn.fit(1, opt=None, cb_funcs=cbfs)
        cb    = learn.find_subcription_by_cls(GetOneBatchCallback)
        return cb.xb, cb.yb

    def adapt_model(self, databunch:DataBunch, normalization:Transform):
        #get rid of norm
        cut   = next( i for i,o in enumerate(self.model.children()) if isinstance(o,nn.AdaptiveAvgPool2d) )
        m_cut = self.model[:cut]
    
        xb,_  = self.getFirstbatch( databunch, normalization )
        pred  = m_cut(xb)
        ni    = pred.shape[1]
    
        self.model = nn.Sequential(
            m_cut, 
            #AdaptiveConcatPool2d(), 
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(ni, databunch.c_out)
            #nn.Linear(ni*2, data.c_out)
        )

class CnnModelManager(ModelManager):

    def initialize(self, is_resnet:bool, uniform:bool=False):
        f       = init.kaiming_uniform_ if uniform else init.kaiming_normal_
        modules = self.find_modules(lambda m: not isinstance(m, nn.Sequential))
        for m in modules:
            if isinstance(m, (nn.Conv2d,nn.Linear) ):
                f(m.weight, a=0.1)
                if getattr(m, 'bias', None) is not None: m.bias.data.zero_()
            if is_resnet and getattr(m, 'bias', None) is not None: m.bias.data.zero_()

"""    
def init_cnn_resnet(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn_resnet(l)

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
