import fire
import time
import rich
import torch
import numpy
import logging
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from sigmafnl.architecture.latin_hypercubes import dataloader

logging.getLogger("lightning").setLevel(logging.ERROR)
torch.set_default_dtype(torch.float32)
torch.manual_seed(23282)

channel_num=16

class net(pl.LightningModule):

    def __init__(self):
        super(net, self).__init__()
        self.conv1 = torch.nn.Conv3d(in_channels=1, out_channels=channel_num, kernel_size=(7,7,7))
        self.conv2 = torch.nn.Conv3d(in_channels=channel_num, out_channels=channel_num, kernel_size=(1,1,1))
        self.conv3 = torch.nn.Conv3d(in_channels=channel_num, out_channels=channel_num, kernel_size=(5,5,5))
        self.conv4 = torch.nn.Conv3d(in_channels=channel_num, out_channels=channel_num, kernel_size=(1,1,1))
        self.conv5 = torch.nn.Conv3d(in_channels=channel_num, out_channels=channel_num, kernel_size=(3,3,3))
        self.conv6 = torch.nn.Conv3d(in_channels=channel_num, out_channels=1, kernel_size=(1,1,1))

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        x = torch.nn.functional.relu(self.conv5(x))
        x = self.conv6(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=4)
        return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                                            "scheduler": scheduler,
                                            "monitor": "loss",
                                    },
                }
        

    def lossfn(self, output, target):
        loss = torch.mean((torch.mean(output, (-4,-3,-2,-1)) - target)**2)
        return loss


    def training_step(self, train_batch, batch_idx):
        snapshot, parameters = train_batch
        output = model(snapshot)
        loss = self.lossfn(output, parameters)
        self.log('loss', loss)
        return {'loss': loss}


model = net().load_from_checkpoint('./lightning_logs/version_20/checkpoints/epoch=148-step=49500.ckpt', map_location='cuda')

def main(channel_num=16, batchsize=1, padding=6, splits=4, basepath=None, checkpoint='checkpoint'):
    
    datalist = dataloader.datalist(basepath=basepath, shuffle=False, start=600, end=800)
    savelist = [str(x[0]).replace('matter', 'sigma8') for x in datalist]
    rich.print(savelist)
    dataset = dataloader.SnapshotDataset(datalist, splits=splits, padding=padding, transform=dataloader.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize)

    torch.set_grad_enabled(False)
    model.eval()
    sigma8_eval = numpy.zeros((512,512,512), dtype=numpy.float32)
    split_size = 512//splits
    cum = []; sigma = []
    for idx, batch in enumerate(data_loader):
        prediction = model(batch[0])
        ii = idx%splits
        cum.append(torch.mean(prediction))#.cpu().detach().numpy()))
        if ii == (splits-1):
            sigma.append([cum, batch[1]])
            rich.print(f'saved {idx//splits}')
            cum = []
        if idx%25 == 24:
            numpy.save('output', sigma)
    '''
    for idx, batch in enumerate(data_loader):
        prediction = model(batch[0])
        ii = idx%splits
         
        if ii < (splits-1):
            sigma8_eval[:,:,ii*split_size:(ii+1)*split_size] = numpy.squeeze(prediction.cpu().detach().numpy())
        elif ii == (splits-1):
            sigma8_eval[:,:,ii*split_size:(ii+1)*split_size] = numpy.squeeze(prediction.cpu().detach().numpy())
            numpy.save(savelist[idx//splits], sigma8_eval)
            sigma8_eval = numpy.zeros((512,512,512), dtype=numpy.float32)
            rich.print(f'saved {idx//splits}')
    '''




if '__main__' == __name__:
    fire.Fire(main)
