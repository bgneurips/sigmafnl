import re
import rich
import fire
import numpy
import torch
import random
import pathlib
from torch.utils.data import Dataset

torch.set_default_dtype(torch.float32)
torch.manual_seed(1342911)


def datalist(basepath=None, shuffle=True):
    
    if not basepath:
         basepath= '/home/ugiri/sigmafnl/datasets/bimodal_dataset/' #s8_p/251/matter_patches_251_512_256_256.npy'
         #basepath= '/scratch/utkarsh/bimodal_dataset/' #s8_p/251/matter_patches_251_512_256_256.npy'
    assert pathlib.Path(basepath), 'Example path not found. Make sure drive is mounted'    
    
    dlist = []
    for i in range(500):
        pname = pathlib.Path(basepath).joinpath(f's8_p/{i}/matter_patches_{i}_512_256_256.npy')
        mname = pathlib.Path(basepath).joinpath(f's8_m/{i}/matter_patches_{i}_512_256_256.npy')
        if mname.exists() and pname.exists():
            dlist.append((pname, 0.849))
            dlist.append((mname, 0.819))
        else:
            print(mname)
    if shuffle: random.shuffle(dlist)
    rich.print(f'Found {len(dlist)} snapshot files')
    return dlist


def circular(x, pad=4):

    x = numpy.concatenate([x, x[0:pad, :, :]], axis=0)
    x = numpy.concatenate([x, x[:, 0:pad, :]], axis=1)
    x = numpy.concatenate([x, x[:, :, 0:pad]], axis=2)
    x = numpy.concatenate([x[-2 * pad:-pad, :, :], x], axis=0)
    x = numpy.concatenate([x[:, -2 * pad:-pad, :], x], axis=1)
    x = numpy.concatenate([x[:, :, -2 * pad:-pad], x], axis=2)

    return x



class SnapshotDataset(Dataset):

    def __init__(self, datalist, voxels=512, splits=2, padding=4, transform=None):
        
        self.datalist = datalist
        self.voxels = voxels
        self.splits = splits
        self.padding = padding
        self.transform = transform

        self.split_size = self.voxels//self.splits
        assert self.voxels/self.splits == self.split_size


    def __len__(self):
        return self.splits*len(self.datalist)

    def __getitem__(self, idx):

        snapshot = numpy.load(self.datalist[idx//self.splits][0])
        snapshot = circular(snapshot, self.padding)
        start = (idx%self.splits)*self.split_size
        sample = {
                'snapshot': snapshot[:, :, start:start+(self.split_size+2*self.padding)], 
                'parameters': self.datalist[idx//self.padding][1]
                 }

        if self.transform:
            sample = self.transform(sample)

        return sample
 
    
class ToTensor(object):

    def __call__(self, sample, ):
        
        snapshot, parameters = sample['snapshot'], sample['parameters']
        return (torch.tensor(snapshot[numpy.newaxis,:]), torch.tensor(parameters))

