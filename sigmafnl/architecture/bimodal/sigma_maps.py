import fire
import rich
import numpy
import torch
import pathlib
from sigmafnl.architecture.bimodal.swnet import net
from sigmafnl.architecture.bimodal.dataloader import datalist, SnapshotDataset, ToTensor

model = net().load_from_checkpoint('../checkpoints/checkpoint_bimodal.ckpt')#, map_location='cpu')

dlist = datalist(shuffle=False)[800:]


slist = [str(x[0]).replace('matter', 'sigma8') for x in dlist]
rich.print(slist)
dataset = SnapshotDataset(dlist, splits=4, transform=ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

voxels = 512
splits = 4
split_size = voxels//splits
sigma8s = []
torch.set_grad_enabled(False)
model.eval()

for idx, batch in enumerate(dataloader):
    prediction = model(batch[0])
    
    if idx%splits == 0:
        rich.print(f'Computing \u03c3\u2088 for {idx//splits}.', end='')
        sigma = numpy.zeros((voxels, voxels, voxels), dtype=numpy.float32)
        sigma[:, :, :split_size] = numpy.squeeze(prediction.cpu().detach().numpy())[:,:,:]
    
    elif idx%splits == (splits-1):
        sigma[:, :, split_size*(splits-1):voxels] = numpy.squeeze(prediction.cpu().detach().numpy())[:,:,:]
        rich.print(' Done. Saving \u03c3\u2088 file')
        sigma8s.append(numpy.mean(sigma))
        numpy.save(slist[idx//splits], sigma)

    else:
        start = split_size*(idx%splits)
        sigma[:, :, start:start+split_size] = numpy.squeeze(prediction.cpu().detach().numpy())[:,:,:]

