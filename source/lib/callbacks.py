from __future__ import annotations
#from .torch import *
from torch import tensor
import torch as torch
import matplotlib.pyplot as plt
#import torch.nn.functional as F

###################### Callbacks #######################
class Callback():
    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')

class CudaCallback(Callback):
    def __init__(self, device): self.device = device
    def begin_fit(  self, e:Event): e.learn.model.to(device)
    def begin_batch(self, e:Event): e.learn.xb, e.learn.yb = e.learn.xb.to(device),e.learn.yb.to(device)

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
    
    def begin_loss(self,e:Event): e.learn.loss = e.learn.loss_func(e.learn.preds, e.learn.yb)

#must always be used for training        
class TrainEvalCallback(Callback):
    def begin_train(self,e:Event): 
        if e.learn.in_train: e.learn.model.train()

    def begin_validate(self, e:Event): e.learn.model.eval()
      
 #must always be used for training        
class ParamScheduler(Callback):
    def __init__(self, pname, sched_funcs): self.pname, self.sched_funcs = pname,sched_funcs

    def begin_fit(self,e:Event):
        #count iteration to adjust the training params to the progress in the training cycle
        self.n_iter = 0
        
        if not isinstance(self.sched_funcs, (list,tuple)):
            self.sched_funcs = [self.sched_funcs] * len(e.learn.opt.param_groups)
            
    def set_param(self, e):
        assert len(e.learn.opt.param_groups)==len(self.sched_funcs)
        fractional_cycle = min(1.,self.n_iter /(e.learn.iters * e.learn.epochs))
        for pg,f in zip(e.learn.opt.param_groups,self.sched_funcs):
            pg[self.pname] = f(fractional_cycle)

    def begin_batch(self,e:Event): 
        if e.learn.in_train: self.set_param(e)       

    def after_batch(self,e:Event): 
        if e.learn.in_train: self.n_iter += 1

#used for alle training
class OptimizerCallback(Callback):
    def begin_step(self, e:Event):
        if e.learn.in_train: e.learn.opt.step()
            
    def after_step(self, e:Event):            
        if e.learn.in_train: e.learn.opt.zero_grad()   

class LR_Finder(Callback):
    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
        self.max_iter,self.min_lr,self.max_lr, self.best_loss = max_iter,min_lr,max_lr, 1e9
        self.alpha = 0.8
        
    def begin_fit(self, e:Event):
        self.n_iter = 0
        self.avg_loss = -1
        self.losses,self.smooth_losses  = [], []
        self.lrs  = [[] for _ in e.learn.opt.param_groups]
        
    def begin_batch(self, e:Event):
        if not e.learn.in_train: return
        pos = self.n_iter/self.max_iter
        lr  = self.min_lr * (self.max_lr/self.min_lr) ** pos
        for pg in e.learn.opt.param_groups: pg['lr'] = lr
        self.n_iter += 1 
        
    def after_batch(self, e:Event):
        if e.learn.in_train:         
            for pg,lr in zip(e.learn.opt.param_groups, self.lrs): lr.append(pg['lr'])
            loss = e.learn.loss.detach().cpu()
            self.losses.append(loss)
            
            if self.avg_loss <0 : self.avg_loss = loss
            else:                 self.avg_loss = self.avg_loss*self.alpha + loss*(1-self.alpha) 
            self.smooth_losses.append( self.avg_loss )
            
            
    def after_step(self, e:Event):
        if not e.learn.in_train: return
        if self.n_iter    >= self.max_iter or e.learn.loss > self.best_loss*10: e.learn.stop = True
        elif e.learn.loss <  self.best_loss: self.best_loss = e.learn.loss
            
    def plot_lr  (self, pgid=-1): 
        fig, ax = plt.subplots()
        plt.plot(self.lrs[pgid], label='learning rate')        
        ax.set(xlabel='iteration', ylabel='learning rate', title='learning rate finder')  
        plt.legend(loc='upper left')
        
    def plot_loss(self, skip_start=0, skip_end=0 ):          
        lrs = self.lrs[-1]
        fig, ax = plt.subplots()
        s = slice(skip_start,-skip_end) if skip_end>0 else slice(skip_start, None)
        l1 = plt.plot(lrs[s], self.losses[s],        label='raw')
        l2 = plt.plot(lrs[s], self.smooth_losses[s], 
                      label=f"smoothed:{self.alpha}: avg_loss*a +(1-{self.alpha})*loss")
        plt.xscale('log')
        ax.set(xlabel='learning rate', ylabel='losses', title='learning rate finder')  
        plt.legend(loc='lower left')


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

class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats, self.valid_stats = AvgStats(metrics,True), AvgStats(metrics,False)

    def begin_epoch(self, e:Event):
        self.train_stats.reset()
        self.valid_stats.reset()

    def after_loss(self, e:Event):
        stats = self.train_stats if e.learn.in_train else self.valid_stats
        with torch.no_grad(): stats.accumulate(e.learn)

    def after_epoch(self, e:Event):
        print(self.train_stats)
        print(self.valid_stats)
        
        
class Recorder(Callback):   
    def begin_fit(self, e:Event):
        self.lrs = [[] for _ in e.learn.opt.param_groups]
        self.losses = []

    def after_batch(self, e:Event):
        if e.learn.in_train:         
            for pg,lr in zip(e.learn.opt.param_groups, self.lrs): lr.append(pg['lr'])
            self.losses.append(e.learn.loss.detach().cpu())

    def plot_lr  (self, pgid=-1): plt.plot(self.lrs[pgid])
    def plot_loss(self):          plt.plot(self.losses)

#######################################   Learner ########################################################            
from enum import Enum,auto
class Stages(Enum):
    begin_fit = auto()
    begin_epoch = auto()
    begin_batch = auto()
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
    
train_batch_stages = [Stages.begin_prediction, Stages.after_prediction,
                      Stages.begin_loss,       Stages.after_loss,
                      Stages.begin_backwards,  Stages.after_backwards,
                      Stages.begin_step,       Stages.after_step ]
valid_batch_stages = [Stages.begin_prediction, Stages.after_prediction,
                      Stages.begin_loss,       Stages.after_loss ]

class Event():
    def __init__(self,learner:Learner):
        self.learn = learner

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
    n_epochs = 0
    epochs   = 0
    loss     = -1
    #private
    _data    = None
    _stop    = None
    
    def __init__(self, model, data, loss_func, opt, cb_funcs):
        self.model,self._data,self.loss_func,self.opt = model,data,loss_func,opt
        self.msn = Messenger()
        self.msn.register_callback_functions(cb_funcs)
        #for cb in listify(cbs): self.msn.register(cb)
        self._stop = False

    def find_subcription_by_cls(self,cls):
        for s in self.msn.subscriptions:
            if type(s) == cls:return s
        
    @property
    def stop(self): return self._stop
    @stop.setter 
    def stop(self, value): 
        if not self._stop : self._stop = value 

    def fit(self, epochs):
        self.epochs, self.loss = epochs, tensor(0.)
        event = Event(self)
        try:
            self.msn.notify(Stages.begin_fit,event)
            for epoch in range(epochs):
                if self.stop: break
                print(f"epoch: {epoch}")
                    
                self.epoch = epoch
                self.msn.notify(Stages.begin_epoch,event)
                
                self.in_train = True
                self.msn.notify(Stages.begin_train,event)
                self.all_batches(train_batch_stages, self._data.train_dl)
                self.msn.notify(Stages.after_train,event)
                self.in_train = False
                
                with torch.no_grad():
                    self.msn.notify(Stages.begin_validate,event)
                    self.all_batches(valid_batch_stages, self._data.valid_dl)
                    self.msn.notify(Stages.after_validate,event)
                        
                self.msn.notify(Stages.after_epoch,event)
        except Exception as e: self.exception_handler(e)
        finally: self.msn.notify(Stages.after_fit,event)
                    
    def all_batches(self, batch_stages, dl):
        ite_count, self.iters = 0, len(dl)
        try:
            for xb,yb in dl: 
                if self.stop: break
                self.one_batch(batch_stages, xb, yb)
                ite_count+=1
                p = (100*ite_count)//self.iters
                if p%20==0 : print(f"{p} %")
        except Exception as e: self.exception_handler(e)
                            
    def one_batch(self, batch_stages, xb, yb):
        event = Event(self)        
        try:
            self.xb,self.yb = xb,yb
            self.msn.notify(Stages.begin_batch,event)
            for msg in batch_stages: 
                self.msn.notify(msg,event)
        except Exception as e: self.exception_handler(e)
        finally: self.msn.notify(Stages.after_batch,event)

    def exception_handler(self, e:Exception ):
        self.stop = True
        import traceback
        print("exception: {e}")
        tb = traceback.format_exc()            
        print(f"exception received 3\n:{tb}")
 