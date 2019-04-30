from lib.data import *
from functools import partial

_m = tensor([0.47, 0.48, 0.45])
_s = tensor([0.29, 0.28, 0.30])
norm_imagenette = partial(normalize, mean=_m.cuda(), std=_s.cuda())
