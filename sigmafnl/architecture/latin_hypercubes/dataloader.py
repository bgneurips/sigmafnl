import re
import rich
import fire
import numpy
import torch
import random
import pathlib
from functools import lru_cache
from torch.utils.data import Dataset

torch.set_default_dtype(torch.float32)
torch.manual_seed(134911)


def datalist(basepath=None, start=0, end=800, shuffle=True):
    
    if not basepath:
         basepath= '/home/ugiri/latin_hypercube_HR_halos/'#251/matter_patches_251_512_256_256.npy'
    assert pathlib.Path(basepath), 'Example path not found. Make sure drive is mounted'    
    
    dlist = []
    for i in range(start, end):
        name = pathlib.Path(basepath).joinpath(f'{i}/nh_{i}_512.npy')
        param = numpy.loadtxt(f'/home/ugiri/latin_hypercube_HR_halos/{i}/Cosmo_params.dat')[-1]
        dlist.append((name, param))
    if shuffle: random.shuffle(dlist)
    rich.print(f'Found {len(dlist)} snapshot files')
    return dlist


def circular(x, pad=6):

    x = numpy.concatenate([x, x[0:pad, :, :]], axis=0)
    x = numpy.concatenate([x, x[:, 0:pad, :]], axis=1)
    x = numpy.concatenate([x, x[:, :, 0:pad]], axis=2)
    x = numpy.concatenate([x[-2 * pad:-pad, :, :], x], axis=0)
    x = numpy.concatenate([x[:, -2 * pad:-pad, :], x], axis=1)
    x = numpy.concatenate([x[:, :, -2 * pad:-pad], x], axis=2)

    return x



class SnapshotDataset(Dataset):

    def __init__(self, datalist, voxels=512, splits=4, padding=8, transform=None):
        
        self.datalist = datalist
        self.voxels = voxels
        self.splits = splits
        self.padding = padding
        self.transform = transform

        self.split_size = self.voxels//self.splits
        assert self.voxels/self.splits == self.split_size

        #self.snapshot_data = numpy.zeros((512,512,512), dtype=numpy.float32)
        #self.old_i = -1


    def __len__(self):
        return self.splits*len(self.datalist)

    def __getitem__(self, idx):

        #self._reader(idx//self.splits)
        snapshot_data = circular(numpy.load(self.datalist[i][0]), self.padding)
        start = (idx%self.splits)*self.split_size
        sample = {
                #'snapshot': self.snapshot_data[:, :, start:start+(self.split_size+2*self.padding)], 
                'snapshot': snapshot_data[:, :, start:start+(self.split_size+2*self.padding)], 
                'parameters': self.datalist[idx//self.splits][1]
                 }

        if self.transform:
            sample = self.transform(sample)

        return sample
    

    def _reader(self, i):
        if self.old_i != i:
            self.old_i = i
            self.snapshot_data = circular(numpy.load(self.datalist[i][0]), self.padding)
 
    
class ToTensor(object):

    def __call__(self, sample, ):
        
        snapshot, parameters = sample['snapshot'], sample['parameters']
        return (torch.tensor(snapshot[numpy.newaxis,:]), torch.tensor(parameters))

