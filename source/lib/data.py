from __future__ import annotations
import torch
from torch import tensor
from torch.utils.data import DataLoader
import pickle
import gzip
import random
from pathlib import Path
from .utilities import *

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


################### file processing ###################
import os
# step back 2 levels in the path and grab the folder name
def grandparent_splitter(fn, valid_name='valid', train_name='train'):
    gp = fn.parent.parent.name
    return True if gp==valid_name else False if gp==train_name else None

class FileList(ItemList):

    @staticmethod
    def _get_files(p, fs, extensions=None):
        p = Path(p)
        res = [p/f for f in fs if not f.startswith('.')
               and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]
        return res
                
    @staticmethod
    def get_files(path, extensions=None, recurse=False, include=None):
        print(path)
        path       = Path(path)
        extensions = setify(extensions)
        extensions = {e.lower() for e in extensions}
        if recurse:
            res = []
            for i,(p,d,f) in enumerate(os.walk(path)): # returns (dirpath, dirnames, filenames)
                if include is not None and i==0: d[:] = [o for o in d if o in include]
                else:                            d[:] = [o for o in d if not o.startswith('.')]
                res += FileList._get_files(p, f, extensions)
            return res
        else:
            f = [o.name for o in os.scandir(path) if o.is_file()]
            return FileList._get_files(path, f, extensions)

    @classmethod
    def from_files(cls, path, extensions, recurse=True, include=None, tfms=None):
        files = FileList.get_files(path, extensions, recurse=recurse, include=include)
        return cls(files, path=path, tfms=tfms)

    @staticmethod
    def split_by_func(ds:FileList, split_condition):
        items = ds.items
        mask = [split_condition(o) for o in items]
        # `None` values will be filtered out
        train = [o for o,m in zip(items,mask) if m==False]
        valid = [o for o,m in zip(items,mask) if m==True ]
        return train,valid

    #split itemnslist in a training and validation list base on a condition
    def split(self,  split_condition):
        train,valid = FileList.split_by_func(self, split_condition)
        return SplitData(self.new(train),self.new(valid))

class SplitData():
    def __init__(self, train, valid): self.train,self.valid = train,valid
        
    def __getattr__(self,k): return getattr(self.train,k)
    #This is needed if we want to pickle SplitData and be able to load it back without recursion errors
    def __setstate__(self,data:Any): self.__dict__.update(data) 

    def __repr__(self): return f'{self.__class__.__name__}\nTrain: {self.train}\nValid: {self.valid}\n'

    def label_data(self, label_function, proc_x=None, proc_y=None):
        self.train = LabeledData.label_by_func(self.train, label_function, proc_x=proc_x, proc_y=proc_y)
        self.valid = LabeledData.label_by_func(self.valid, label_function, proc_x=proc_x, proc_y=proc_y)

    def get_train_dataLoader(self, batch_size:int, num_workers:int, shuffle:bool=True):
        return DataLoader(self.train, batch_size=batch_size,  num_workers=num_workers, shuffle=shuffle)
    def get_valid_dataLoader(self, batch_size:int, num_workers:int):
        return DataLoader(self.valid, batch_size=batch_size,  num_workers=num_workers)

########################  labelling of categorical data  ########################
"""
Labeling has to be done *after* splitting, because it uses *training* set information to apply to 
the *validation* set. In order to **convert label strings to numbers** we create a vocab with a list 
of possible labels in the training set, and then convert them to numbers based on this *vocab*.
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

def parent_labeler(fn): return fn.parent.name

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
    def label_by_func(cls, x, f, proc_x=None, proc_y=None):
        y = ItemList( [f(o) for o in x.items], path=x.path)
        return cls(x, y, proc_x=proc_x, proc_y=proc_y)



#################### Transform ##############################
def view_tfm(*size):
    def _inner(x): return x.view(*((-1,)+size))
    return _inner

class Transform(): 
    pass

#########################################################
########################  VISION  ########################
#########################################################

import PIL,mimetypes
image_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))
class ImageList(FileList):
    def get(self, fn): return PIL.Image.open(fn)

    #def from_files(cls, path, extensions=image_extensions, recurse=True, include=None, tfms=None):
    @classmethod
    def from_files(cls, path, extensions=image_extensions, recurse=True, include=None, tfms=None):
        return super(ImageList,cls).from_files(path, extensions, recurse=recurse, include=include, tfms=tfms )

#################### functional transforms ##############################
def normalize(x, mean, std):
    return (x-mean[...,None,None]) / std[...,None,None]
def denormalize(x, mean, std):
    return ( (x * std[...,None,None]) + mean[...,None,None])

def make_rgb(item): return item.convert('RGB')

# to_byte_tensor convert the image to a byte tensor 
def to_byte_tensor(item):
    res = torch.ByteTensor(torch.ByteStorage.from_buffer(item.tobytes()))
    w,h = item.size
    return res.view(h,w,-1).permute(2,0,1)

#convert a byte tensor to float and scales to [0-1] by dividing by 255
def to_float_tensor(item): return item.float().div_(255.)

class MakeRGB(Transform):
    def __call__(self, item): return item.convert('RGB')

class PilRandomFlip(Transform):
    def __init__(self, p=0.5): self.p=p
    def __call__(self, x):
        return x.transpose(PIL.Image.FLIP_LEFT_RIGHT) if random.random()<self.p else x

class PilRandomDihedral(Transform):
    def __init__(self, p=0.75): self.p=p
    def __call__(self, x):
        #-1:  is the identity transform
        #0-6: is FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM, ROTATE_90, ROTATE_180, ROTATE_270, TRANSPOSE, TRANSVERSE
        tf = random.randint(-1,6)
        return x if tf==-1 else x.transpose(tf)

class ResizeFixed(Transform):
    def __init__(self,size, resize_tfm:int=PIL.Image.BILINEAR):
        if isinstance(size,int): size =(size,size)
        self.size = size
        self.resize_tfm = resize_tfm

    def __call__(self, item): return item.resize(self.size, self.resize_tfm)

    
class GeneralCrop(Transform):
    @staticmethod
    def process_sz(sz):
        sz = listify(sz)
        return tuple(sz if len(sz)==2 else [sz[0],sz[0]])
    def __init__(self, size, crop_size=None, resample=PIL.Image.BILINEAR): 
        self.resample,self.size = resample,GeneralCrop.process_sz(size)
        self.crop_size = None if crop_size is None else GeneralCrop.process_sz(crop_size)
        
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
import math
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


################### top is refactored from the code below this line #####################


#def normalize_to(train, valid):
#    m,s = train.mean(), train.std()
#    def normalize(x, m, s): return (x-m)/s
#    return normalize(train, m, s), normalize(valid, m, s)

#This is a slightly different from what was seen during the lesson,
#   we'll discuss the changes in lesson 11
"""
First, let's define the processor. 
We also define a `ProcessedItemList` with an `obj` method that can get 
the unprocessed items: for instance a processed label will be an index between 0 and the number of classes - 1, 
the corresponding `obj` will be the name of the class. The first one is needed by the model for the training, 
but the second one is better for displaying the objects.
"""
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