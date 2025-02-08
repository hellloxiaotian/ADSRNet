import os

from data import common
from data import srdata

import numpy as np

import torch
import torch.utils.data as data

class Benchmark(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(Benchmark, self).__init__(
            args, name=name, train=train, benchmark=True
        )
    def _set_filesystem(self, dir_data):
        if self.name in ['Set5', 'Set14']:
            self.apath = os.path.join(dir_data, self.name, 'original')
            self.dir_hr = self.apath
            self.dir_lr = os.path.join(dir_data, self.name, 'LRbicx2' + str(self.scale[0]))
            print('self.dir_lr:', self.dir_lr)
            self.ext = ('.png', '.png')
        elif self.name in ['B100']:
            self.apath = os.path.join(dir_data, 'B100', 'HR')
            self.dir_hr = self.apath
            self.dir_lr = os.path.join(dir_data, 'B100', 'bicubic_x' + str(self.scale[0]))
            print('self.dir_lr:', self.dir_lr)
            self.ext = ('.png', '.png')
        elif self.name in ['Urban100']:
            # self.apath = os.path.join(dir_data, 'U100', 'HR')
            self.apath = os.path.join(dir_data, 'Urban100', 'HR')
            self.dir_hr = self.apath
            # self.dir_lr = os.path.join(dir_data, 'U100', 'bicubic_x' + str(self.scale[0]))
            self.dir_lr = os.path.join(dir_data, 'Urban100', 'bicubic_x' + str(self.scale[0]))
            print('self.dir_lr:', self.dir_lr)
            self.ext = ('.png', '.png')
        elif self.name in ['WHU']:
            self.apath = os.path.join(dir_data, 'WHU', 'HR')
            self.dir_hr = self.apath
            # self.dir_lr = os.path.join(dir_data, 'U100', 'bicubic_x' + str(self.scale[0]))
            self.dir_lr = os.path.join(dir_data, 'WHU', 'LR' + str(self.scale[0]))
            print('self.dir_lr:', self.dir_lr)
        elif self.name in ['RSS']:
            self.apath = os.path.join(dir_data, 'RSS', 'HR')
            self.dir_hr = self.apath
            # self.dir_lr = os.path.join(dir_data, 'U100', 'bicubic_x' + str(self.scale[0]))
            self.dir_lr = os.path.join(dir_data, 'RSS', 'LR' + str(self.scale[0]))
            print('self.dir_lr:', self.dir_lr)
        elif self.name in ['Day365']:
            self.apath = os.path.join(dir_data, 'Day365', 'HR')
            self.dir_hr = self.apath
            # self.dir_lr = os.path.join(dir_data, 'U100', 'bicubic_x' + str(self.scale[0]))
            self.dir_lr = os.path.join(dir_data, 'Day365', 'LR' + str(self.scale[0]))
            print('self.dir_lr:', self.dir_lr)
        else:
            print('Not benchmark!')
            return
