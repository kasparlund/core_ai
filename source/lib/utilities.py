import os
import requests
import gzip, pickle
from torch import tensor

from fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import format_time
from .datasets import *

from collections import Iterable
from collections.abc import Iterable

####################  containers  ##################
class ListContainer():
    def __init__(self, items): self.items = items
    def __getitem__(self, idx):
        if isinstance(idx, (int,slice)): return self.items[idx]
        if isinstance(idx[0],bool):
            assert len(idx)==len(self) # bool mask
            return [o for m,o in zip(idx,self.items) if m]
        return [self.items[i] for i in idx]
    def __len__(self): return len(self.items)
    def __iter__(self): return iter(self.items)
    def __setitem__(self, i, o): self.items[i] = o
    def __delitem__(self, i): del(self.items[i])
    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'
        if len(self)>10: res = res[:-1]+ '...]'
        return res
        
"""
We use the `ListContainer` to store our objects in an `ItemList`. 
The `get` method will need to be subclassed to explain how to access an element 
(open an image for instance), then the private `_get` method can allow us to apply any 
additional transform to it.
`new` will be used in conjunction with `__getitem__` (that works for one index or a list of indices) 
to create training and validation set from a single stream when we split the data.

Transforms only need to be functions that take an element of the ItemList and transform it. 
If they need state, they can be defined as a class. Also, having them as a class allows 
to define an _order attribute (default 0) that is used to sort the transforms.

"""
from pathlib import Path
class ItemList(ListContainer):
    def __init__(self, items, path='.', tfms=None):
        super().__init__(items)
        self.path,self.tfms = Path(path),tfms

    def __repr__(self): return f'{super().__repr__()}\nPath: {self.path}'
    def new(self, items): return self.__class__(items, self.path, tfms=self.tfms)

    def  get(self, i): return i
    def _get(self, i): return compose(self.get(i), self.tfms)

    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        if isinstance(res,list): return [self._get(o) for o in res]
        return self._get(res)

def setify(o): return o if isinstance(o,set) else set(listify(o))

def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]

from collections import OrderedDict
def uniqueify(x, sort=False):
    res = list(OrderedDict.fromkeys(x).keys())
    if sort: res.sort()
    return res

####################  operations  ##################
def compose(x, funcs, *args, order_key='_order', **kwargs):
    key = lambda o: getattr(o, order_key, 0)
    for f in sorted(listify(funcs), key=key): x = f(x, **kwargs)
    return x

####################  IO  ##################
def download_url(url:str, dest:str, overwrite:bool=False, pbar:progress_bar=None,
                 show_progress=True, chunk_size=1024*1024, timeout=4, retries=5)->None:
    "Download `url` to `dest` unless it exists and not `overwrite`."
    if os.path.exists(dest) and not overwrite: return

    s = requests.Session()
    s.mount('http://',requests.adapters.HTTPAdapter(max_retries=retries))
    u = s.get(url, stream=True, timeout=timeout)
    try: file_size = int(u.headers["Content-Length"])
    except: show_progress = False

    with open(dest, 'wb') as f:
        nbytes = 0
        if show_progress: pbar = progress_bar(range(file_size), auto_update=False, leave=False, parent=pbar)
        try:
            for chunk in u.iter_content(chunk_size=chunk_size):
                nbytes += len(chunk)
                if show_progress: pbar.update(nbytes)
                f.write(chunk)
        except requests.exceptions.ConnectionError as e:
            fname = url.split('/')[-1]
            from datasets import Config
            data_dir = Config().data_path()
            timeout_txt =(f'\n Download of {url} has failed after {retries} retries\n'
                          f' Fix the download manually:\n'
                          f'$ mkdir -p {data_dir}\n'
                          f'$ cd {data_dir}\n'
                          f'$ wget -c {url}\n'
                          f'$ tar -zxvf {fname}\n\n'
                          f'And re-run your code once the download is successful\n')
            print(timeout_txt)
            import sys;sys.exit(1)

################################  Processor baseclass  ##################################
class Processor(): 
    def process(self, items): return items

################################  load picled trainings data  ##################################
def load_pickled_train_valid_data(path:Path):
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    return map(tensor, (x_train,y_train,x_valid,y_valid))



########################################  Test  ########################################
import operator
def test(a,b,cmp,cname=None):
    if cname is None: cname=cmp.__name__
    assert cmp(a,b),f"{cname}:\n{a}\n{b}"
def test_eq(a,b): test(a,b,operator.eq,'==')
def near(a,b): return torch.allclose(a, b, rtol=1e-3, atol=1e-5)
def test_near(a,b): test(a,b,near)
#tests of setify
"""
test_eq(setify('aa'), {'aa'})
test_eq(setify(['aa',1]), {'aa',1})
test_eq(setify(None), set())
test_eq(setify({1}), {1})
test_eq(setify(1), {1})
"""
