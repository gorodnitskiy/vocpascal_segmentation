import torch
import pytorch_lightning as pl


def loss_custom(pred, target):
    return torch.nn.functional.cross_entropy(
        pred[(target != 255).unsqueeze(1).expand(pred.shape)],
        target[target != 255])


class IoU(pl.metrics.Metric):
    def __init__(self, n_classes=21):
        super().__init__()
        self.n_classes = n_classes
        self.add_state("inter", default=torch.zeros([21]), dist_reduce_fx='sum')
        self.add_state("union", default=torch.zeros([21]), dist_reduce_fx='sum')

    def update(self, preds, target):
        res = preds.argmax(dim=1)
        for index in range(self.n_classes):
            truth = (target.cpu() == index)
            preds = (res == index)

            inter = truth.logical_and(preds.cpu())
            union = truth.logical_or(preds.cpu())

            self.inter[index] += inter.float().sum()
            self.union[index] += union.float().sum()

    def compute(self):
        return self.inter.sum() / self.union.sum()


class VOCPascalNet(pl.LightningModule):
    def __init__(self, model, loss, iou_metric=IoU()):
        super().__init__()
        self.model = model
        self.iou_metric = iou_metric
        self.loss = loss

    def forward(self, input):
        return self.model(input)

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch['image'])
        loss_val = self.loss(pred, batch['seg'])

        acc_val = self.iou_metric(pred, batch['seg'])
        self.log('IoU/train', acc_val, on_epoch=True)
        self.log('loss/train', loss_val, on_epoch=True)
        return loss_val

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch['image'])
        loss_val = self.loss(pred, batch['seg'])
        acc_val = self.iou_metric(pred, batch['seg'])

        self.log('loss/valid', loss_val, on_epoch=True)
        self.log('IoU/valid', acc_val, on_epoch=True)
        return loss_val

    def configure_optimizers(self):
        self.optim = torch.optim.Adam(self.parameters(), lr=1.0e-3)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, 'min')
        return self.optim

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        epoch = args[0]
        batch_idx = args[1]
        val_accuracy = self.trainer.logged_metrics['IoU/valid']
        if epoch != 0 and batch_idx == 0:
            self.sched.step(val_accuracy)
