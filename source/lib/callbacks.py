from __future__ import annotations
from collections.abc import Iterable
from torch import tensor
import torch as torch
import matplotlib.pyplot as plt
import time
from fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import format_time

from .utilities import *
from .model import *


###################### Callbacks #######################
class Callback():
    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')

#generalize this callback to debug callback
class GetOneBatchCallback(Callback):
    def after_preprocessing(self, e:Event): 
        self.xb,self.yb = e.learn.xb,e.learn.yb
        e.learn.stop = True

class CudaCallback(Callback):
    def __init__(self, device): self.device = device
    def begin_fit(  self, e:Event): e.learn.model.to(self.device)
    def begin_batch(self, e:Event): e.learn.xb, e.learn.yb = e.learn.xb.to(self.device),e.learn.yb.to(self.device)

class SimpleCudaCallback(Callback):
    def __init__(self, device): super()(device = torch.device('cuda',0))
    
        
class BatchTransformXCallback(Callback):
    def __init__(self, tfm): self.tfm = tfm
    def begin_batch(self, e:Event): e.learn.xb = self.tfm(e.learn.xb)

        
#must always be used for training        
class TrainableModelCallback(Callback):
    def begin_prediction(self,e:Event): e.learn.preds = e.learn.model(e.learn.xb)

    def begin_backwards(self,e:Event): 
        if e.learn.in_train: e.learn.loss.backward()
    
    def begin_loss(self,e:Event): 
        e.learn.loss = e.learn.loss_func(e.learn.preds, e.learn.yb)

#must always be used for training        
class TrainEvalCallback(Callback):
    def begin_train(self,e:Event): 
        if e.learn.in_train: e.learn.model.train()

    def begin_validate(self, e:Event): e.learn.model.eval()
      
 
def accuracy(out, yb): return (torch.argmax(out, dim=1)==yb).float().mean()

class AvgStats():
    def __init__(self, metrics, in_train): self.metrics,self.in_train = metrics,in_train

    def reset(self):
        self.tot_loss,self.count = 0.,0
        self.tot_mets = [0.] * len(self.metrics)

    @property
    def all_stats(self): return [self.tot_loss.item()] + self.tot_mets
    @property
    def avg_stats(self): return [o/self.count for o in self.all_stats]

    def __repr__(self):
        if not self.count: return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"

    def accumulate(self, learn):
        bn = learn.xb.shape[0]
        self.tot_loss += learn.loss * bn
        self.count += bn
        for i,m in enumerate(self.metrics):
            self.tot_mets[i] += m(learn.preds, learn.yb) * bn

"""
class DebugCallback(Callback):
    def __init__(self, cb_name, f_cond=None): self.cb_name,self.f_cond = cb_name,f_cond
    def __call__(self, cb_name):
        if cb_name==self.cb_name:
            if self.f_cond: self.f_cond(self.run)
            else:      set_trace()
"""

###################################### Hooks ###################################### 
from functools import partial
class Hook():
    def __init__(self, layer, func): 
        self.layer, self.hook = layer, layer.register_forward_hook(partial(func, self))
    def remove(self): self.hook.remove()
    def __del__(self): self.remove()


class Hooks(ListContainer):
    def __init__(self, modules, f): 
        super().__init__([Hook(m, f) for m in modules])
            
    def __enter__(self, *args): return self
    def __exit__ (self, *args): self.remove()
    def __del__(self): self.remove()

    def __delitem__(self, i):
        self[i].remove()
        super().__delitem__(i)
        
    def remove(self):
        for h in self: h.remove()

class HookCallback(Callback):   
    def __init__(self, hookProcessor): self.hookProcessor = hookProcessor

    def begin_fit(self, e:Event):
        self.lrs = [[] for _ in e.learn.opt.param_groups]
        self.losses = []

    def after_batch(self, e:Event):
        if e.learn.in_train:         
            for pg,lr in zip(e.learn.opt.param_groups, self.lrs): lr.append(pg['lr'])
            self.losses.append(e.learn.loss.detach().cpu())


def append_stats(hook, module, inp, outp, max_activation=5):
    if module.training:
        if not hasattr(hook,'stats'): hook.stats = ([],[],[])
        means,stds,hists = hook.stats
        means.append(outp.data.mean().cpu())
        stds .append(outp.data.std().cpu())
        hists.append(outp.data.cpu().histc(100,-max_activation,max_activation)) #histc isn't implemented on the GPU


class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats, self.valid_stats = AvgStats(metrics,True), AvgStats(metrics,False)
        self.first=True

    def begin_epoch(self, e:Event):        
        self.train_stats.reset()
        self.valid_stats.reset()
        self.start_time = time.time()
        if self.first:
            met_names = ['loss'] + [m.__name__ for m in self.train_stats.metrics]
            names = ['epoch'] + [f'train_{n}' for n in met_names] + [
                    f'valid_{n}' for n in met_names] + ['time']
            e.learn.logger(names)
            self.first = False

    def after_loss(self, e:Event):
        stats = self.train_stats if e.learn.in_train else self.valid_stats
        with torch.no_grad(): stats.accumulate(e.learn)

    def after_epoch(self, e:Event):
        #print(self.train_stats)
        #print(self.valid_stats)
        stats = [str(e.learn.epoch)] 
        for o in [self.train_stats, self.valid_stats]:
            stats += [f'{v:.6f}' for v in o.avg_stats] 
        stats += [format_time(time.time() - self.start_time)]
        e.learn.logger(stats)
        
from IPython.display import display, Javascript
from fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import format_time
class ProgressCallback(Callback):
    def begin_fit(self,e:Event):
        self.mbar = master_bar(range(e.learn.epochs))
        self.mbar.on_iter_begin()
        e.learn.logger = partial(self.mbar.write, table=True)
        
    def after_fit(self,e:Event): self.mbar.on_iter_end()
    def after_batch(self,e:Event): self.pb.update(e.learn.iter)
    def begin_train   (self,e:Event): self.set_pb(e.learn)
    def begin_validate(self,e:Event): self.set_pb(e.learn)
    def set_pb(self,learn:Learner):
        self.pb = progress_bar(learn.dl, parent=self.mbar, auto_update=False)
        self.mbar.update(learn.epoch)        
        
class Recorder(Callback):   
    def begin_fit(self, e:Event):
        self.lrs,self.train_losses, self.valid_losses = [],[],[]
        self.optimizers = None
        self.epochs = e.learn.epochs        
        
    def begin_epoch(self, e:Event):        
        self.valid_losses.append(0)
        self.valid_iterations = 0
            
    def after_batch(self, e:Event):
        if e.learn.in_train:         
            if self.optimizers is None : 
                self.optimizers = { k:[v] for k,v in e.learn.opt.getOptimizers().items() }
            else:                        
                for k,v in e.learn.opt.getOptimizers().items(): self.optimizers[k].append(v)            
            self.train_losses.append(e.learn.loss.detach().cpu())
        else:
            self.valid_losses[e.learn.epoch] += e.learn.loss.detach().cpu() * e.learn.xb.shape[0]
            self.valid_iterations            += e.learn.xb.shape[0]
            
    def after_epoch(self, e:Event):  
        if self.valid_iterations > 0:
            self.valid_losses[e.learn.epoch] /= self.valid_iterations
            
    def plot_lr(self):
        fig, ax = plt.subplots()
        for k,v in self.optimizers.items(): ax.plot(v,label=k)
        ax.legend(loc='upper left')
        ax.set(xlabel='iteration', ylabel='optimizer', title='optimizers')  
            
    def plot_loss(self, skip_start=0, skip_end=0 ): 
        #resample validation losses so the slicin works
        fig, ax = plt.subplots()
        s           = slice(skip_start,-skip_end) if skip_end>0 else slice(skip_start, None)
        ticksize    = int(len(self.train_losses)/self.epochs)
        tick_labels = [i for i in range(1,self.epochs+1)]
        tick_pos    = [i*ticksize for i in tick_labels]
        l1 = ax.plot(list(range(len(self.train_losses)))[s],self.train_losses[s],label="training")
        l2 = ax.plot(tick_pos[s],self.valid_losses[s],label="validation")
        plt.xticks(tick_pos[s],tick_labels[s])    
        ax.set(xlabel='epochs', ylabel="losses")  
        ax.legend()
        
#######################################   Learner ########################################################            
class Event():
    def __init__(self,learner:Learner):
        self.learn = learner

from enum import Enum,auto
class Stages(Enum):
    begin_fit = auto()
    begin_epoch = auto()
    begin_batch = auto()
    begin_preprocessing = auto(),
    after_preprocessing = auto(),
    begin_prediction = auto()
    after_prediction = auto()
    begin_loss = auto()
    after_loss = auto()
    begin_backwards = auto()
    after_backwards = auto()
    begin_step = auto()
    after_step = auto()
    after_batch = auto()
    after_epoch = auto()
    after_fit = auto()
    begin_validate = auto()
    after_validate = auto()  
    begin_train = auto()
    after_train = auto()  
    
train_batch_stages = [Stages.begin_preprocessing, Stages.after_preprocessing, 
                      Stages.begin_prediction, Stages.after_prediction,
                      Stages.begin_loss,       Stages.after_loss,
                      Stages.begin_backwards,  Stages.after_backwards,
                      Stages.begin_step,       Stages.after_step ]
valid_batch_stages = [Stages.begin_preprocessing, Stages.after_preprocessing,
                      Stages.begin_prediction, Stages.after_prediction,
                      Stages.begin_loss,       Stages.after_loss ]

class Messenger():
    subscriptions = None    
    def __init__(self): self.subscriptions = []
        
    # cbs is instances of Callback whereas cb_funcs are functions that initializes a Callback 
    def register( self, cb:Callback ): self.subscriptions.append(cb)
        
    def register_callback_functions( self, cb_funcs ):
        for cbf in cb_funcs: self.register(cbf())
        
    def notify(self, msg:Stages, event):
        for cb in self.subscriptions: 
            f = getattr(cb, msg.name, None)
            if f is not None and not event.learn.stop: 
                #print(f"in_train: {event.learn.in_train } callback: {type(cb).__name__}.{msg.name}")            
                f(event)
            if event.learn.stop:break
                
#are we missing a begin_train that match begin_validate??????
class Learner():
    #public
    model    = None
    opt      = None
    xb       = None
    yb       = None
    in_train = False
    #epoch    = 0
    epochs   = 0
    loss     = -1
    #private
    _data    = None
    _stop    = None
    
    def __init__(self, model, data, loss_func):
        self.model,self._data,self.loss_func = model,data,loss_func
        #for cb in listify(cbs): self.msn.register(cb)
        self._stop = False
        self.logger = print

    def find_subcription_by_cls(self,cls):
        for s in self.msn.subscriptions:
            if type(s) == cls:return s
        
    @property
    def stop(self): return self._stop
    @stop.setter 
    def stop(self, value): 
        print("stop")
        if not self._stop : self._stop = value 

    def fit(self, epochs, opt, cb_funcs):
        self.epochs, self.loss = epochs, tensor(0.)

        self.msn = Messenger()
        self.msn.register_callback_functions(cb_funcs)
        self.opt = opt

        event = Event(self)
        try:
            self.msn.notify(Stages.begin_fit, event)
            for epoch in range(epochs):
                if self.stop: break
                    
                self.epoch = epoch  #due to progressbar
                self.msn.notify(Stages.begin_epoch,event)
                
                self.in_train = True
                self.all_batches(self._data.train_dl, train_batch_stages, 
                                 Stages.begin_train, Stages.after_train)
                self.in_train = False
                
                with torch.no_grad():
                    self.all_batches(self._data.valid_dl, 
                                     valid_batch_stages, 
                                     Stages.begin_validate,
                                     Stages.after_validate)
                        
                self.msn.notify(Stages.after_epoch,event)
        except Exception as e: self.exception_handler(e)
        finally: self.msn.notify(Stages.after_fit, event)

        self.epoch, self.in_train = 0, False
                    
    def all_batches(self, dl, batch_stages, begin_msg:Stages, after_msg:Stages):
        event = Event(self)        
        self.dl    = self._data.train_dl #due to progress bar
        self.iters = len(dl)

        try:
            self.msn.notify(begin_msg,event)
            for i,(xb,yb) in enumerate(dl): 
                if self.stop: break
                self.iter = i
                self.one_batch(batch_stages, xb, yb)
        except Exception as e: self.exception_handler(e)
        finally: self.msn.notify(after_msg,event)
        self.dl = None
             
    def one_batch(self, batch_stages, xb, yb):
        event = Event(self)        
        self.xb,self.yb = xb,yb
        try:
            self.msn.notify(Stages.begin_batch,event)
            for msg in batch_stages: 
                self.msn.notify(msg,event)
        except Exception as e: self.exception_handler(e)
        finally: self.msn.notify(Stages.after_batch,event)
        self.xb,self.yb = None,None

    def exception_handler(self, e:Exception ):
        self.stop = True
        import traceback
        print("exception: {e}")
        tb = traceback.format_exc()            
        print(f"exception received 3\n:{tb}")
 