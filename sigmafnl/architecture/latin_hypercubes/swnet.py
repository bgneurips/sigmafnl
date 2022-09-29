import fire
import time
import torch
import numpy
import logging
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from sigmafnl.architecture.latin_hypercubes import dataloader

logging.getLogger("lightning").setLevel(logging.ERROR)
torch.set_default_dtype(torch.float32)
torch.manual_seed(23282)

channel_num=8

class net(pl.LightningModule):

    def __init__(self, learning_rate=5e-5):
        super(net, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.conv1 = torch.nn.Conv3d(in_channels=1, out_channels=channel_num, kernel_size=(7,7,7))
        self.bn1 = torch.nn.BatchNorm3d(channel_num)
        self.conv2 = torch.nn.Conv3d(in_channels=channel_num, out_channels=channel_num, kernel_size=(5,5,5))
        self.bn2 = torch.nn.BatchNorm3d(channel_num)
        self.conv3 = torch.nn.Conv3d(in_channels=channel_num, out_channels=channel_num, kernel_size=(5,5,5))
        self.bn3 = torch.nn.BatchNorm3d(channel_num)
        self.conv4 = torch.nn.Conv3d(in_channels=channel_num, out_channels=channel_num, kernel_size=(3,3,3))
        self.bn4 = torch.nn.BatchNorm3d(channel_num)
        self.conv5 = torch.nn.Conv3d(in_channels=channel_num, out_channels=1, kernel_size=(1,1,1))

    def forward(self, x):
        x = self.bn1(torch.nn.functional.leaky_relu(self.conv1(x)))
        x = self.bn2(torch.nn.functional.leaky_relu(self.conv2(x)))
        x = self.bn3(torch.nn.functional.leaky_relu(self.conv3(x)))
        x = self.bn4(torch.nn.functional.leaky_relu(self.conv4(x)))
        x = self.conv5(x)
        return x

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
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
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}


model = net()

def main(channel_num=8, batchsize=1, padding=6, splits=4, basepath=None, checkpoint='checkpoint'):
    
    datalist = dataloader.datalist(basepath=basepath, start=0, end=800)
    dataset = dataloader.SnapshotDataset(datalist, splits=splits, padding=padding, transform=dataloader.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, num_workers=8, shuffle=True)


    trainer = pl.Trainer(gpus=1, strategy=DDPPlugin(find_unused_parameters=False), max_epochs=100, 
            log_every_n_steps=800, precision=16, accumulate_grad_batches=1)
    trainer.fit(model, data_loader)
    trainer.save_checkpoint(f"{checkpoint}_{int(time.time())}.ckpt")


if '__main__' == __name__:
    fire.Fire(main)
