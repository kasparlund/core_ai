from lib.callbacks import *
            
def maybe_update(os, dest, f):
    for o in os:
        for k,v in f(o).items():
            if k not in dest: dest[k] = v

def get_defaults(d): return getattr(d,'_defaults',{})            

class Optimizer():
    def __init__(self, params, steppers, **defaults):
        self.steppers = listify(steppers)
        maybe_update(self.steppers, defaults, get_defaults)
        # might be a generator
        self.param_groups = list(params)
        # ensure params is a list of lists
        if not isinstance(self.param_groups[0], list): self.param_groups = [self.param_groups]
        self.hypers = [{**defaults} for p in self.param_groups]

    def grad_params(self):
        return [(p,hyper) for pg,hyper in zip(self.param_groups,self.hypers)
            for p in pg if p.grad is not None]

    def zero_grad(self):
        for p,hyper in self.grad_params():
            p.grad.detach_()
            p.grad.zero_()

    def step(self):
        for p,hyper in self.grad_params(): compose(p, self.steppers, **hyper)

def sgd_step(p, lr, **kwargs):
    p.data.add_(-lr, p.grad.data)
    return p
sgd_step._defaults = dict(lr=0.3)

def weight_decay(p, lr, wd, **kwargs):
    p.data.mul_(1 - lr*wd)
    return p
weight_decay._defaults = dict(wd=0.)

def l2_reg(p, lr, wd, **kwargs):
    p.grad.data.add_(wd, p.data)
    return p
l2_reg._defaults = dict(wd=0.)

def momentum_step(p, lr, grad_avg, **kwargs):
    p.data.add_(-lr, grad_avg)
    return p


#used for alle training
class OptimizerCallback(Callback):
    def begin_step(self, e:Event):
        if e.learn.in_train: e.learn.opt.step()
            
    def after_step(self, e:Event):            
        if e.learn.in_train: e.learn.opt.zero_grad()   

class Recorder(Callback):   
    def begin_fit(self, e:Event):
        self.lrs,self.losses = [],[]

    def after_batch(self, e:Event):
        if e.learn.in_train:         
            #for pg,lr in zip(e.learn.opt.param_groups, self.lrs): lr.append(pg['lr'])
            self.lrs.append( e.learn.opt.hypers[-1]['lr'] )
            self.losses.append(e.learn.loss.detach().cpu())

    def plot_lr(self):   plt.plot(self.lrs)
    def plot_loss(self): plt.plot(self.losses)

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

##############################   StatefulOptimizer  ############################## 

#########  something is not working -maybe order?? ##################
class StatefulOptimizer(Optimizer):
    def __init__(self, params, steppers, stats=None, **defaults): 
        self.stats = listify(stats)
        maybe_update(self.stats, defaults, get_defaults)
        super().__init__(params, steppers, **defaults)
        self.state = {}
        
    def step(self):
        for p,hyper in self.grad_params():
            if p not in self.state:
                #Create a state for p and call all the statistics to initialize it.
                self.state[p] = {}
                maybe_update(self.stats, self.state[p], lambda o: o.init_state(p))
            state = self.state[p]
            for stat in self.stats: state = stat.update(p, state, **hyper)
            compose(p, self.steppers, **state, **hyper)
            self.state[p] = state

class Stat():
    _defaults = {}
    def init_state(self, p): raise NotImplementedError
    def update(self, p, state, **kwargs): raise NotImplementedError    


#In Adam, we use the gradient averages but with dampening (not like in SGD with momentum), so let's add this to the `AverageGrad` class.        
class AverageGrad(Stat):
    _defaults = dict(mom=0.9)
    
    def __init__(self, dampening:bool=False): self.dampening=dampening
    def init_state(self, p): return {'grad_avg': torch.zeros_like(p.grad.data)}
    def update(self, p, state, mom, **kwargs):
        state['mom_damp'] = 1-mom if self.dampening else 1.
        state['grad_avg'].mul_(mom).add_(state['mom_damp'], p.grad.data)
        return state
    
#We also need to track the moving average of the gradients squared.
class AverageSqrGrad(Stat):
    _defaults = dict(sqr_mom=0.99)
    
    def __init__(self, dampening:bool=True): self.dampening=dampening
    def init_state(self, p): return {'sqr_avg': torch.zeros_like(p.grad.data)}
    def update(self, p, state, sqr_mom, **kwargs):
        state['sqr_damp'] = 1 - sqr_mom if self.dampening else 1.
        state['sqr_avg'].mul_(sqr_mom).addcmul_(state['sqr_damp'], p.grad.data, p.grad.data)
        return state
    
#We will also need the number of steps done during training for the debiasing.
class StepCount(Stat):
    def init_state(self, p): return {'step': 0}
    def update(self, p, state, **kwargs):
        state['step'] += 1
        return state

#################################   Adam   #################################
#This helper function computes the debias term. If we dampening, damp = 1 - mom 
#and we get the same result as before. If we don't use dampening, (damp = 1) 
#we will need to divide by 1 - mom because that term is missing everywhere.
def debias(mom, damp, step): return damp * (1 - mom**step) / (1-mom)

#Adam step with debias term for the exponential averageing:
def adam_step(p, lr, mom, mom_damp, step, sqr_mom, sqr_damp, grad_avg, sqr_avg, eps, **kwargs):
    debias1 = debias(mom,     mom_damp, step)
    debias2 = debias(sqr_mom, sqr_damp, step)
    p.data.addcdiv_(-lr / debias1, grad_avg, (sqr_avg/debias2).sqrt() + eps)
    return p
adam_step._defaults = dict(eps=1e-5)

#################################    LAMB   #################################
def lamb_step(p, lr, mom, mom_damp, step, sqr_mom, sqr_damp, grad_avg, sqr_avg, eps, wd, **kwargs):
    debias1 = debias(mom,     mom_damp, step)
    debias2 = debias(sqr_mom, sqr_damp, step)
    r1 = p.data.pow(2).mean().sqrt()
    step = (grad_avg/debias1) / ((sqr_avg/debias2).sqrt()+eps) + wd*p.data
    r2 = step.pow(2).mean().sqrt()
    p.data.add_(-lr * min(r1/r2,10), step)
    return p
lamb_step._defaults = dict(eps=1e-6, wd=0.)

lamb = partial(StatefulOptimizer, steppers=[lamb_step], 
               stats=[AverageGrad(dampening=True), AverageSqrGrad(), StepCount()])
                   
############################## Optimizer combinations  ############################## 
sgd_wd_opt    = partial(Optimizer, steppers=[weight_decay, sgd_step], wd=0.01)
sgd_wd_l2_opt = partial(Optimizer, steppers=[weight_decay, sgd_step], wd=0.01)
sgd_mom_wd_opt = partial(StatefulOptimizer,steppers=[momentum_step,weight_decay], 
                         stats=[AverageGrad()], wd=0.01)

#problems with both adam and lamb
adam_opt = partial(StatefulOptimizer, steppers=[adam_step], 
                   stats=[AverageGrad(dampening=True), AverageSqrGrad(), StepCount()])
