import matplotlib.pyplot as plt
import math
import numpy as np
from .callbacks import *

def show_image(im, ax=None, figsize=(3,3)):
    if ax is None: _,ax = plt.subplots(1, 1, figsize=figsize)
    ax.axis('off')
    ax.imshow(im.permute(1,2,0))

def show_batch(x, c=4, r=None, figsize=None):
    n = len(x)
    if r is None: r = int(math.ceil(n/c))
    if figsize is None: figsize=(c*3,r*3)
    fig,axes = plt.subplots(r,c, figsize=figsize)
    for xi,ax in zip(x,axes.flat): show_image(xi, ax)

##################layer stats####################

"""
def get_min(h,pct_lower_bins):
    h1     = torch.stack(h.stats[2]).t().float()
    n_bins = h1.shape[0]
    idx    = int(round(pct_lower_bins/100*n_bins) +1)
    return (h1[:idx].sum(0)/h1.sum(0)*100).numpy().astype(np.int)

def get_min2(h,threshold):
    h1     = torch.stack(h.stats[2]).t().float()
    n_bins = h1.shape[0]
    idx    = int(round(pct_lower_bins/100*n_bins) +1)
    return (h1[:idx].sum(0)/h1.sum(0)*100).numpy().astype(np.int)
"""

def plot_layer_stats( hooks:Hooks, pct_lower_bins = 2 ):
    def get_hist(h): return torch.stack(h.stats[2]).t().float().log1p()

    rows = int( len(hooks)/2 + 0.5)
    fig,axes = plt.subplots(rows,2, figsize=(15,3*rows))
    for i,ax,h in zip(range(len(hooks)),axes.flatten(), hooks):
        ax.imshow(get_hist(h), origin='lower', aspect="auto", interpolation="bicubic")
        ax.set(xlabel='iterations', ylabel="histogram", title=f"l:{i}, {type(h.layer)}: ln(output + 1)")  
        #ax.set_axis_off()
        ax.get_yaxis().set_visible(False)
        plt.axis('off')
    plt.tight_layout()
    

    """
    fig,axes = plt.subplots(rows,2, figsize=(15,3*rows))
    sds = np.fromiter( h.stats[2] for h in hooks, dtype=np.float32, count=len(hooks))
    sds_mean = np.mean(sds)

    for i,ax,h in zip(range(len(hooks)),axes.flatten(), hooks):
        ax.plot( get_min2(h, 0.1 * sds_mean) )
        #ax.plot( get_min(h,pct_lower_bins) )
        ax.set_ylim(0,100)
        ax.set(xlabel='iterations', ylabel="% near zero",  title=f"layer {i}, {type(h.layer)}: output near zero")  
    plt.tight_layout()    
    """

    fig,(ax0,ax1) = plt.subplots(1,2, figsize=(15,6))
    for h in hooks:
        ms,ss = h.stats[:2]
        ax0.plot(ms)
        ax0.set(xlabel='iterations', ylabel="mean activation",  title=f"mean of activations pr layers")  
        ax0.legend(range(len(hooks)));
        ax1.plot(ss)
        ax1.set(xlabel='iterations', ylabel="std activation",  title=f"std of activations pr layers")  
        ax1.legend(range(len(hooks)));
    plt.tight_layout()     
