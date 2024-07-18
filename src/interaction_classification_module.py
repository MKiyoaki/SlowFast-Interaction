import torch
import torch.nn as nn
import torch.nn.functional as F


class InteractionClassificationModule(nn.Module):
    def __init__(self, model, loss_fn=F.cross_entropy):
        super(InteractionClassificationModule, self).__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        y_hat = self.model(batch["video"])
        loss = self.loss_fn(y_hat, batch["label"])

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.model(batch["video"])
        loss = self.loss_fn(y_hat, batch["label"])

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)

