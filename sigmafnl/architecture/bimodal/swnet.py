import fire
import time
import torch
import numpy
import logging
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from sigmafnl.architecture.bimodal import dataloader

logging.getLogger("lightning").setLevel(logging.ERROR)
torch.set_default_dtype(torch.float32)
torch.manual_seed(232882)

channel_num=16

class net(pl.LightningModule):

    def __init__(self):
        super(net, self).__init__()
        self.conv1 = torch.nn.Conv3d(in_channels=1, out_channels=channel_num, kernel_size=(5,5,5))
        self.conv2 = torch.nn.Conv3d(in_channels=channel_num, out_channels=channel_num, kernel_size=(1,1,1))
        self.conv3 = torch.nn.Conv3d(in_channels=channel_num, out_channels=channel_num, kernel_size=(3,3,3))
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                                            "scheduler": scheduler,
                                            "monitor": "loss",
                                    },
                }
        
        #return optimizer

    def lossfn(self, output, target):
        loss = torch.mean((torch.mean(output, (-4,-3,-2,-1)) - target)**2)
        return loss


    def training_step(self, train_batch, batch_idx):
        snapshot, parameters = train_batch
        output = model(snapshot)
        loss = self.lossfn(output, parameters)
        self.log('loss', loss) 
        return {'loss': loss}


model = net()

def main(channel_num=16, batchsize=1, padding=4, basepath=None, checkpoint='checkpoint'):
    
    datalist = dataloader.datalist(basepath=basepath)[:900]
    dataset = dataloader.SnapshotDataset(datalist, padding=padding, transform=dataloader.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, num_workers=8)


    trainer = pl.Trainer(gpus=1, strategy=DDPPlugin(find_unused_parameters=False),
                            log_every_n_steps=450, max_epochs=100, precision=16, accumulate_grad_batches=2)
    trainer.fit(model, data_loader)
    trainer.save_checkpoint(f"{checkpoint}_{int(time.time())}.ckpt")


if '__main__' == __name__:
    fire.Fire(main)
