import fire
import rich
import numpy
import torch
import pathlib
from sigmafnl.architecture.bimodal.swnet import net
from sigmafnl.architecture.bimodal.dataloader import datalist, SnapshotDataset, ToTensor
#from sigmafnl.architecture.latin_hypercubes.dataloader import datalist, SnapshotDataset, ToTensor

#model = net().load_from_checkpoint('../checkpoints/checkpoint_bimodal.ckpt')#, map_location='cpu')
#model = net().load_from_checkpoint('/home/ugiri/bimodal/checkpoint_channel_num_16_voxels_512_with_patching_smaller_patch.ckpt')#, map_location='cpu')
#model = net().load_from_checkpoint('/home/ugiri/bimodal/lightning_logs/version_32/checkpoints/epoch=20-step=8399.ckpt')#, map_location='cpu')
model = net().load_from_checkpoint('/home/ugiri/bimodal/lightning_logs/version_34/checkpoints/epoch=131-step=90911.ckpt')#, map_location='cpu')

#dlist = datalist(shuffle=False, start=0, end=200)[:]
dlist = datalist(shuffle=False, basepath='/scratch/utkarsh/bimodal_dataset/')[:100]

#dlist = [pathlib.Path(f'/home/ugiri/bimodalfnl250/fnl250seed{i}/matter_patches_{i}_512_256_256.npy') for i in range(48,49)]
#dlist = [x for x in dlist if x.exists()]
#dlist = [(x, 0.849) for x in dlist]

#dlist = [pathlib.Path(f'/scratch/utkarsh/fiducial/{i}/matter_patches_{i}_512_512_512.npy') for i in range(10000, 10100)]
#dlist = [pathlib.Path(f'/home/ugiri/bimodalfnl250/fnl250seed{i}/matter_patches_{i}_512_256_256.npy') for i in range(2,10)]
#dlist = [x for x in dlist if x.exists()]
#dlist = [(x, 0.834) for x in dlist]

rich.print(dlist)
print(len(dlist))
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
        #numpy.save(slist[idx//splits], sigma)

    else:
        start = split_size*(idx%splits)
        sigma[:, :, start:start+split_size] = numpy.squeeze(prediction.cpu().detach().numpy())[:,:,:]

numpy.save('sigma8s', sigma8s)
