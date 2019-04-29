
from __future__ import annotations
import lib.datasets as datasets
import pickle
import gzip
from torch import tensor
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
import PIL,mimetypes,os
from pathlib import Path

#import callbacks  #due to listitem
from lib.callbacks import *

#import PIL,os,mimetypes


##################   DATA ressources #################
#from lib.datasets import *
def get_mnist_data(path:Path):
    #MNIST_URL='http://deeplearning.net/data/mnist/mnist.pkl'
    #path = datasets.download_data(MNIST_URL, ext='.gz')
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    return map(tensor, (x_train,y_train,x_valid,y_valid))


def normalize_chan(x, mean, std):
    return (x-mean[...,None,None]) / std[...,None,None]

_m = tensor([0.47, 0.48, 0.45])
_s = tensor([0.29, 0.28, 0.30])
norm_imagenette = partial(normalize_chan, mean=_m.cuda(), std=_s.cuda())

##################   DATA  #################

class Dataset():
    def __init__(self, x, y): self.x,self.y = x,y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i],self.y[i]
        
class DataBunch():
    def __init__(self, train_dl, valid_dl, c_in, c_out):
        self.train_dl,self.valid_dl,self.c_in,self.c_out = train_dl,valid_dl,c_in,c_out

    @property
    def train_ds(self): return self.train_dl.dataset

    @property
    def valid_ds(self): return self.valid_dl.dataset


def normalize(x, m, s): return (x-m)/s

def normalize_to(train, valid):
    m,s = train.mean(),train.std()
    return normalize(train, m, s), normalize(valid, m, s)

def make_rgb(item): return item.convert('RGB')

#################### Transform ##############################
class Transform(): _order=0


# to_byte_tensor convert the image to a byte tensor 
def to_byte_tensor(item):
    res = torch.ByteTensor(torch.ByteStorage.from_buffer(item.tobytes()))
    w,h = item.size
    return res.view(h,w,-1).permute(2,0,1)
to_byte_tensor._order=20

#convert a byte tensor to float and scales to [0-1] by dividing by 255
def to_float_tensor(item): return item.float().div_(255.)
to_float_tensor._order=30


class PilTransform(Transform): _order=11

class PilRandomFlip(PilTransform):
    def __init__(self, p=0.5): self.p=p
    def __call__(self, x):
        return x.transpose(PIL.Image.FLIP_LEFT_RIGHT) if random.random()<self.p else x

class PilRandomDihedral(PilTransform):
    def __init__(self, p=0.75): self.p=p
    def __call__(self, x):
        #-1:  is the identity transform
        #0-6: is FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM, ROTATE_90, ROTATE_180, ROTATE_270, TRANSPOSE, TRANSVERSE
        tf = random.randint(-1,6)
        return x if tf==-1 else x.transpose(tf)

from random import randint

class ResizeFixed(Transform):
    _order=10
    def __init__(self,size, resize_tfm:int=PIL.Image.BILINEAR):
        if isinstance(size,int): size =(size,size)
        self.size = size
        self.resize_tfm = resize_tfm

    def __call__(self, item): return item.resize(self.size, self.resize_tfm)

def process_sz(sz):
    sz = listify(sz)
    return tuple(sz if len(sz)==2 else [sz[0],sz[0]])
class GeneralCrop(PilTransform):
    def __init__(self, size, crop_size=None, resample=PIL.Image.BILINEAR): 
        self.resample,self.size = resample,process_sz(size)
        self.crop_size = None if crop_size is None else process_sz(crop_size)
        
    def default_crop_size(self, w,h): pass #return default_crop_size(w,h)

    def __call__(self, x):
        csize = self.default_crop_size(*x.size) if self.crop_size is None else self.crop_size
        #return x.transform(self.size, PIL.Image.EXTENT, self.get_corners(*x.size, *csize), resample=self.resample)
        return x.resize(self.size, resample=self.resample, box=self.get_corners(*x.size, *csize))
    
    def get_corners(self, w, h): return (0,0,w,h)

class CenterCrop(GeneralCrop):
    def __init__(self, size, scale=1.14, resample=PIL.Image.BILINEAR):
        super().__init__(size, resample=resample)
        self.scale = scale
        
    def default_crop_size(self, w,h): return [w/self.scale,h/self.scale]
    
    def get_corners(self, w, h, wc, hc):
        return ((w-wc)//2, (h-hc)//2, (w-wc)//2+wc, (h-hc)//2+hc)

"""
This is the usual data augmentation used on ImageNet (introduced [here](https://arxiv.org/pdf/1409.4842.pdf)) 
that consists of selecting 8 to 100% of the image area and a scale between 3/4 and 4/3 as a crop, 
then resizing it to the desired size. 
It combines some zoom and a bit of squishing at a very low computational cost.
"""
import random
class RandomResizedCrop(GeneralCrop):
    def __init__(self, size, scale=(0.08,1.0), ratio=(3./4., 4./3.), resample=PIL.Image.BILINEAR):
        super().__init__(size, resample=resample)
        self.scale,self.ratio = scale,ratio
    
    def default_crop_size(self, w,h): 
        return [w/self.scale[0],h/self.scale[1]]

    def get_corners(self, w, h, wc, hc):
        area = w*h
        #Tries 10 times to get a proper crop inside the image.
        for attempt in range(10):
            area = random.uniform(*self.scale) * area
            ratio = math.exp(random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))
            new_w = int(round(math.sqrt(area * ratio)))
            new_h = int(round(math.sqrt(area / ratio)))
            if new_w <= w and new_h <= h:
                left = random.randint(0, w - new_w)
                top  = random.randint(0, h - new_h)
                return (left, top, left + new_w, top + new_h)
        
        # Fallback to central crop
        left,top = randint(0,w-self.crop_size[0]),randint(0,h-self.crop_size[1])
        return (left, top, left+self.crop_size[0], top+self.crop_size[1])


################### IO ###################
from pathlib import Path

def setify(o): return o if isinstance(o,set) else set(listify(o))

def _get_files(p, fs, extensions=None):
    p = Path(p)
    res = [p/f for f in fs if not f.startswith('.')
           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]
    return res
                
def get_files(path, extensions=None, recurse=False, include=None):
    path = Path(path)
    extensions = setify(extensions)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i,(p,d,f) in enumerate(os.walk(path)): # returns (dirpath, dirnames, filenames)
            if include is not None and i==0: d[:] = [o for o in d if o in include]
            else:                            d[:] = [o for o in d if not o.startswith('.')]
            res += _get_files(p, f, extensions)
        return res
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        return _get_files(path, f, extensions)

####################  Preprocessing of data  ####################
"""
Labeling has to be done *after* splitting, because it uses *training* set information to apply to 
the *validation* set, using a *Processor*.

A *Processor* is a transformation that is applied to all the inputs once at initialization, 
with some *state* computed on the training set that is then applied without modification on 
the validation set (and maybe the test set or at inference time on a single item). 
For instance, it could be **processing texts** to **tokenize**, then **numericalize** them. 
In that case we want the validation set to be numericalized with exactly the same vocabulary as the training set.

Another example is in **tabular data**, where we **fill missing values** with (for instance) 
the median computed on the training set. That statistic is stored in the inner state of the *Processor* 
and applied on the validation set.

In our case, we want to **convert label strings to numbers** in a consistent and reproducible way. 
So we create a list of possible labels in the training set, and then convert our labels to numbers 
based on this *vocab*.
"""

class Processor(): 
    def process(self, items): return items

class CategoryProcessor(Processor):
    def __init__(self): self.vocab=None
    
    def __call__(self, items):
        #The vocab is defined on the first use.
        if self.vocab is None:
            self.vocab = uniqueify(items)
            self.otoi  = {v:k for k,v in enumerate(self.vocab)}
        return [self.proc1(o) for o in items]
    def proc1(self, item):  return self.otoi[item]
    
    def deprocess(self, idxs):
        assert self.vocab is not None
        return [self.deproc1(idx) for idx in idxs]
    def deproc1(self, idx): return self.vocab[idx]


#####################  IMAGE  #####################

# Our filenames are `path` object, we can find the directory of the file with `.parent`.
# We need to go back two folders before since the last folders are the class names.
def grandparent_splitter(fn, valid_name='valid', train_name='train'):
    gp = fn.parent.parent.name
    return True if gp==valid_name else False if gp==train_name else None

def split_by_func(ds:ItemList, f):
    items = ds.items
    mask = [f(o) for o in items]
    # `None` values will be filtered out
    train = [o for o,m in zip(items,mask) if m==False]
    valid = [o for o,m in zip(items,mask) if m==True ]
    return train,valid

class SplitData():
    def __init__(self, train, valid): self.train,self.valid = train,valid
        
    def __getattr__(self,k): return getattr(self.train,k)
    #This is needed if we want to pickle SplitData and be able to load it back without recursion errors
    def __setstate__(self,data:Any): self.__dict__.update(data) 
    
    @classmethod
    def split_by_func(cls, il:ItemList, f):
        lists = map(il.new, split_by_func(il, f))
        return cls(*lists)

    def __repr__(self): return f'{self.__class__.__name__}\nTrain: {self.train}\nValid: {self.valid}\n'




########################  Transform  ########################
class Transform(): _order=0


########################  vision  ########################
import PIL,os,mimetypes

image_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))

class MakeRGB(Transform):
    def __call__(self, item): return item.convert('RGB')

def make_rgb(item): return item.convert('RGB')

image_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))
class ImageList(ItemList):
    @classmethod
    def from_files(cls, path, extensions=None, recurse=True, include=None, **kwargs):
        if extensions is None: extensions = image_extensions
        return cls(get_files(path, extensions, recurse=recurse, include=include), path, **kwargs)

    def get(self, fn): return PIL.Image.open(fn)


########################  labelling of categorical data  ########################

class Processor(): 
    def process(self, items): return items

"""
Here we label according to the folders of the images, so simply fn.parent.name. 
We label the training set first with a newly created CategoryProcessor so that it computes its 
inner vocab on that set. Then we label the validation set using the same processor, which means it 
uses the same vocab. The end result is another SplitData object.
"""
class CategoryProcessor(Processor):
    def __init__(self): self.vocab=None
    
    def __call__(self, items):
        #The vocab is defined on the first use.
        if self.vocab is None:
            self.vocab = uniqueify(items)
            self.otoi  = {v:k for k,v in enumerate(self.vocab)}
        return [self.proc1(o) for o in items]
    def proc1(self, item):  return self.otoi[item]
    
    def deprocess(self, idxs):
        assert self.vocab is not None
        return [self.deproc1(idx) for idx in idxs]
    def deproc1(self, idx): return self.vocab[idx]

"""
First, let's define the processor. We also define a `ProcessedItemList` with an `obj` method that can get 
the unprocessed items: for instance a processed label will be an index between 0 and the number of classes - 1, 
the corresponding `obj` will be the name of the class. The first one is needed by the model for the training, 
but the second one is better for displaying the objects.
"""
def parent_labeler(fn): return fn.parent.name

def _label_by_func(ds, f, cls=ItemList): return cls([f(o) for o in ds.items], path=ds.path)

#This is a slightly different from what was seen during the lesson,
#   we'll discuss the changes in lesson 11
"""        
class ProcessedItemList(ListContainer):
    def __init__(self, inputs, processor):
        self.processor = processor
        items = processor.process(inputs)
        super().__init__(items)

    def obj(self, idx):
        res = self[idx]
        if isinstance(res,(tuple,list,Generator)): return self.processor.deprocess(res)
        return self.processor.deproc1(idx)
class LabeledData():
    def __init__(self, x, y): self.x,self.y = x,y

    def __repr__(self): return f'{self.__class__.__name__}\nx: {self.x}\ny: {self.y}\n'
    def __getitem__(self,idx): return self.x[idx],self.y[idx]
    def __len__(self): return len(self.x)

    @classmethod
    def label_by_func(cls, il, f, proc=None):
        labels = _label_by_func(il, f)
        proc_labels = ProcessedItemList(labels, proc)
        return cls(il, proc_labels)
"""        
class LabeledData():
    def process(self, il, proc): return il.new(compose(il.items, proc))

    def __init__(self, x, y, proc_x=None, proc_y=None):
        self.x,self.y = self.process(x, proc_x),self.process(y, proc_y)
        self.proc_x,self.proc_y = proc_x,proc_y
        
    def __repr__(self): return f'{self.__class__.__name__}\nx: {self.x}\ny: {self.y}\n'
    def __getitem__(self,idx): return self.x[idx],self.y[idx]
    def __len__(self): return len(self.x)
    
    def x_obj(self, idx): return self.obj(self.x, idx, self.proc_x)
    def y_obj(self, idx): return self.obj(self.y, idx, self.proc_y)
    
    def obj(self, items, idx, procs):
        isint = isinstance(idx, int) or (isinstance(idx,torch.LongTensor) and not idx.ndim)
        item = items[idx]
        for proc in reversed(listify(procs)):
            item = proc.deproc1(item) if isint else proc.deprocess(item)
        return item

    @classmethod
    def label_by_func(cls, il, f, proc_x=None, proc_y=None):
        d = _label_by_func(il, f)
        return cls(il, d, proc_x=proc_x, proc_y=proc_y)

def label_train_valid_data(sd, label_function, proc_x=None, proc_y=None):
    train = LabeledData.label_by_func(sd.train, label_function, proc_x=proc_x, proc_y=proc_y)
    valid = LabeledData.label_by_func(sd.valid, label_function, proc_x=proc_x, proc_y=proc_y)
    return SplitData(train,valid)        

import math
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
