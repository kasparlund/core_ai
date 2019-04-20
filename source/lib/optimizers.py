from lib.callbacks import *
from typing import *

class OptimizerFunction():
    "abstract class to implement optimization of the models parameters and acces the current optimizer parameters"
    def __init__(self, sched_func): 
        self.sched_func, self.optimizers,  = sched_func, None

    def getOptimizers(self): return self.optimizers
    
    def update(self, progress:float): 
        "progress: is the percentage progres in the planned epochs and iterations"
        self.optimizers = self.updateOptimizers(progress)

    def optimize(self, params:Collection[torch.nn.Parameter], mov_avg:torch.nn.Parameter=None): 
        "update the parameters. mov_avg may is None, not None when using OptimizerCallback, StatefulOptimizer respectively"
        raise NotImplementedError("def optimize Must be implemented")

    def updateOptimizers(self, progress:float): 
        raise NotImplementedError("def getOptimizers: Must be implemented")
    
class OptimizerCallback(Callback):
    def begin_fit(self,e:Event):
        #count iteration to adjust the training params to the progress in the training cycle
        self.n_iter  = 0      
        self.params  = [ p for p in e.learn.model.parameters() if p.requires_grad ]
        #self.mov_avg = [ p*0 for p in self.params ]

    def begin_batch(self,e:Event): 
        if e.learn.in_train:
            self.fractional_cycle = min(1.,self.n_iter /(e.learn.iters * e.learn.epochs))
            e.learn.opt.update(self.fractional_cycle)
        
    def begin_step(self, e:Event):
        if e.learn.in_train:
            #for p in self.params: e.learn.opt.optimize(p)
            e.learn.opt.optimize(self.params)
            #self.mom = 0.9    
            #self.mov_avg = self.mov_avg*self.mom + (1-self.mom) *p.grad.data
            #state['grad_avg'].mul_(mom).add_(state['mom_damp'], p.grad.data)
           
    def after_step(self, e:Event):            
        if e.learn.in_train:
            for p in self.params:
                p.grad.detach_()
                p.grad.zero_()
                
    def after_batch(self,e:Event): 
        if e.learn.in_train: self.n_iter += 1

class LR_Finder(Callback):
    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10, beta = 0.8):
        self.max_iter,self.min_lr,self.max_lr, self.best_loss = max_iter,min_lr,max_lr, 1e9
        self.beta = beta

    def begin_fit(self, e:Event):
        self.n_iter = 0
        self.mov_avg = -1
        self.losses,self.smooth_losses  = [], []
        self.lrs  = []
        #self.lrs  = [[] for _ in e.learn.opt.param_groups]
        
    def after_loss(self, e:Event):
        if not e.learn.in_train: return
        pos = self.n_iter/self.max_iter
        e.learn.opt.lr = self.min_lr * (self.max_lr/self.min_lr) ** pos
        
    def after_batch(self, e:Event):
        if e.learn.in_train:         
            loss         = e.learn.loss.item()
            self.mov_avg = loss if self.mov_avg < 0 else self.mov_avg*self.beta + loss*(1-self.beta) 
            self.losses.append(loss)
            self.smooth_losses.append( self.mov_avg )            
            self.lrs.append(e.learn.opt.lr)
            self.n_iter += 1 
            
    def after_step(self, e:Event):
        if not e.learn.in_train: return
        if self.n_iter >= self.max_iter or self.mov_avg > 4.0*self.best_loss:
            e.learn.stop = True
        if e.learn.loss <  self.best_loss: self.best_loss = e.learn.loss.item()
            
    def plot_lr  (self, pgid=-1): 
        fig, ax = plt.subplots()
        plt.plot(self.lrs, label='learning rate')        
        ax.set(xlabel='iteration', ylabel='learning rate', title='learning rate finder')  
        plt.legend(loc='upper left')
        
    def plot_loss(self, skip_start=0, skip_end=0 ):          
        fig, ax = plt.subplots()
        s  = slice(skip_start,-skip_end) if skip_end>0 else slice(skip_start, None)
        l1 = plt.plot(self.lrs[s], self.losses[s], label='raw')
        l2 = plt.plot(self.lrs[s], self.smooth_losses[s], 
                      label=f"smoothed:{self.beta}: mov_avg*a +(1-{self.beta})*loss")
        plt.xscale('log')
        ax.set(xlabel='learning rate', ylabel='losses', title='learning rate finder')  
        plt.legend(loc='lower left')        

class SGD(OptimizerFunction):
    "sgd with momentum and weight decay"
    #params = params - learning_rate * params_grad - learning_rate * wd * params"
    def __init__(self,sched_func, max_lr=0.3, max_wd=0.0): 
        super().__init__(sched_func)
        self.lr, self.max_lr = max_lr, max_lr
        self.wd, self.max_wd = max_wd, max_wd
    def updateOptimizers(self, progress:float):
        self.lr = self.max_lr*self.sched_func(progress)
        self.wd = self.max_wd*self.sched_func(progress) if self.max_wd>0 else 0
        return {"lr":self.lr,"wd":self.wd}
    def optimize(self, params:Collection[torch.nn.Parameter], mov_avg:torch.nn.Parameter=None):
        for p in params: 
            if self.wd > 0.: p.data.add_(-self.lr*self.wd,p.data)
            p.data.add_(-self.lr, p.grad.data)
            
class AverageGrad():
    def __init__(self): self.avg = None
    def update(self, mom, params):
        #mean_avg, mean_grad = 0.,0.
        if self.avg is None: 
            self.avg = [p.grad.data.clone() for p in params]
        else:   
            for avg,p in zip(self.avg,params) :
                avg.mul_(mom).add_(1-mom, p.grad.data)
                #avg[:] = (self.mom * avg + (1 - self.mom ) * p.grad.data)[:]
            #    mean_avg  += avg.abs().mean()
            #    mean_grad += p.grad.data.abs().mean()
        #print(f"mean_abs_avg, mean_ab_p: {mean_avg:.4f}, {mean_grad:.4f}" )
        
class AverageSqrGrad():
    def __init__(self): 
        self.avg = None
        self.sqr_mom = 0.99

    def update(self, mom, params):
        if self.avg is None: 
            #no debiase that the first avg is set to 100% of the first squared gradient
            self.avg = [p.grad.data.pow(2) for p in params]
        else:   
            for avg,p in zip(self.avg,params) :
                avg.mul_(self.sqr_mom).addcmul_(1-self.sqr_mom, p.grad.data, p.grad.data)

class SGD_Momentum(OptimizerFunction):
    "sgd with momentum and weight decay"
    #mov_avg = momentum*mov_avg +(1-momentum) * params_grad
    #params  = params - learning_rate * mov_avg - learning_rate * wd * params_grad
    def __init__(self,sched_func, max_lr=0.3, moms=(0.85,0.95), max_wd=0.): 
        super().__init__(sched_func)
        self.lr,  self.max_lr     = max_lr, max_lr
        self.mom, self.moms_range = moms[1], moms
        self.wd,  self.max_wd     = max_wd, max_wd
        self.avg_grad = AverageGrad()
    def updateOptimizers(self, progress:float):
        self.lr  = self.max_lr*self.sched_func(progress)
        self.mom = self.moms_range[0] + (self.moms_range[1]-self.moms_range[0])*self.sched_func(progress)
        self.wd  = self.max_wd*self.sched_func(progress) if self.max_wd>0 else 0
        return {"lr":self.lr,"mom":self.mom,"wd":self.wd}
    def optimize(self, params:Collection[torch.nn.Parameter], mov_avg:torch.nn.Parameter=None):
        
        self.avg_grad.update(self.mom,params)
                
        for mov_avg,p in zip(self.avg_grad.avg,params) :
            if self.wd > 0.: p.data.mul_(1-self.lr*self.wd)
            p.data.add_(-self.lr, mov_avg)

class Adam(OptimizerFunction):
    #wd as in Decoupled Weight Decay Regularization: https://arxiv.org/abs/1711.05101
    #momentum_1, momentum_2, eps are typically 0.9, 0.99, 1e-8
    #avg     = momentum_1*avg     +(1-momentum_1) * params_grad
    #sqr_avg = momentum_2*avg +(1-momentum_2) * params_grad * params_grad
    #params  = params - learning_rate * avg / (sqrt(sqr_avg)+eps)
    def __init__(self,sched_func, max_lr=0.3, moms=(0.85,0.95), max_wd=0.): 
        super().__init__(sched_func)
        self.lr,  self.max_lr     = max_lr, max_lr
        self.mom, self.moms_range = moms[1], moms
        self.wd,  self.max_wd     = max_wd, max_wd
        self.eps = 1e-8
        self.avg_grad     = AverageGrad()
        self.avg_sqr_grad = AverageSqrGrad()
    def updateOptimizers(self, progress:float):
        self.lr  = self.max_lr*self.sched_func(progress)
        self.mom = self.moms_range[0] + (self.moms_range[1]-self.moms_range[0])*self.sched_func(progress)
        self.wd  = self.max_wd
        #self.wd  = self.max_wd*(1-min(1,self.sched_func(progress))) if self.max_wd>0 else 0
        return {"lr":self.lr,"mom":self.mom,"wd":self.wd}
    def optimize(self, params:Collection[torch.nn.Parameter], avg:torch.nn.Parameter=None):
        self.avg_grad.update(self.mom,params)
        self.avg_sqr_grad.update(self.mom,params)
                
        for p,avg_grad,avg_sqr_grad in zip(params,self.avg_grad.avg,self.avg_sqr_grad.avg) :
            #p.data.add_(-self.lr, avg_grad/(avg_sqr_grad.sqrt()+self.eps))
            if self.wd > 0.: p.data.mul_(1-self.lr*self.wd)
            p.data.addcdiv_(-self.lr, avg_grad, avg_sqr_grad.sqrt().add_(self.eps) )
                        

class LAMB(OptimizerFunction):
    #momentum_1, momentum_2, eps are typically 0.9, 0.99, 1e-8
    #avg     = momentum_1*avg     +(1-momentum_1) * params_grad
    #sqr_avg = momentum_2*avg +(1-momentum_2) * params_grad * params_grad
    #params  = params - learning_rate * avg / (sqrt(sqr_avg)+eps)
    def __init__(self,sched_func, max_lr=0.3, moms=(0.85,0.95), max_wd=0.): 
        super().__init__(sched_func)
        self.lr,  self.max_lr     = max_lr, max_lr
        self.mom, self.moms_range = 0.5*(moms[0]+moms[1]), moms
        self.wd,  self.max_wd     = max_wd, max_wd
        self.eps = 1e-9
        self.avg_grad     = AverageGrad()
        self.avg_sqr_grad = AverageSqrGrad()
    def updateOptimizers(self, progress:float):
        #self.lr  = self.max_lr*self.sched_func(progress)
        #self.mom = self.moms_range[0] + (self.moms_range[1]-self.moms_range[0])*self.sched_func(progress)
        #self.wd  = self.max_wd
        #self.wd  = self.max_wd*(1-min(1,self.sched_func(progress))) if self.max_wd>0 else 0
        return {"lr":self.lr,"mom":self.mom,"wd":self.wd}
    def optimize(self, params:Collection[torch.nn.Parameter], avg:torch.nn.Parameter=None):
        self.avg_grad.update(self.mom,params)
        self.avg_sqr_grad.update(self.mom,params)

        for p,avg_grad,avg_sqr_grad in zip(params,self.avg_grad.avg,self.avg_sqr_grad.avg) :
            r1   = p.data.pow(2).mean().sqrt() #magnitude of parameters: square-root of mean squared parameters
            
            step = avg_grad /(avg_sqr_grad.sqrt()+self.eps)
            if self.wd>0 : step.add_(self.wd,p.data)
                
            r2 = step.pow(2).mean().sqrt()     #magnitude of gradients: square-root of mean squared gradients
            
            p.data.add_(-self.lr * min(r1/(r2+self.eps),10), step)



"""
#must always be used for training        
class ParamScheduler(Callback):
    def __init__(self, pname, sched_func): self.pname, self.sched_func = pname,sched_func

    def begin_fit(self,e:Event):
        #count iteration to adjust the training params to the progress in the training cycle
        self.n_iter = 0
            
    def begin_batch(self,e:Event): 
        if e.learn.in_train: 
            for h in e.learn.opt.hypers:
                fractional_cycle = min(1.,self.n_iter /(e.learn.iters * e.learn.epochs))
                h[self.pname] = self.sched_func(fractional_cycle)

    def after_batch(self,e:Event): 
        if e.learn.in_train: self.n_iter += 1
"""