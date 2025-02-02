import os, sys

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import ImageFolder

class FixedCrop(torch.nn.Module):
    def __init__(self, bApply = True):
        super().__init__()
        self.bApply = bApply

    def forward(self, img):
        if (self.bApply):
            return transforms.functional.crop(img, 324, 406, 490, 860)
        else:
            return img

# Here we define a new class to turn the ResNet model that we want to use as a feature extractor
# into a pytorch-lightning module so that we can take advantage of lightning's Trainer object.
# We aim to make it a little more general by allowing users to define the number of prediction classes.
class ResNetClassifier(pl.LightningModule):
    resnets = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }
    optimizers = {"adam": Adam, "sgd": SGD}

    def __init__(
        self,
        num_classes,
        resnet_version,
        dataset_root,
        optimizer="adam",
        lr=1e-3,
        batch_size=16,
        transfer=True,
        tune_fc_only=True,
        crop_input=False,
        gray_input=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.dataset_root = dataset_root
        self.lr = lr
        self.batch_size = batch_size

        self.optimizer = self.optimizers[optimizer]
        # instantiate loss criterion
        self.loss_fn = (
            nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
        )
        # create accuracy metric
        self.acc = Accuracy(
            task="binary" if num_classes == 1 else "multiclass", num_classes=num_classes
        )
        # Using a pretrained ResNet backbone
        self.resnet_model = self.resnets[resnet_version](weights="DEFAULT")
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features
        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Linear(linear_size, num_classes)
        self.crop_input = crop_input
        self.gray_input = gray_input

        if tune_fc_only:  # option to only tune the fully-connected layers
            for child in list(self.resnet_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

        self.output_preds = []
        self.output_gts = []


    def forward(self, X):
        return self.resnet_model(X)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def _step(self, batch):
        x, y = batch
        preds = self(x)

        if self.num_classes == 1:
            preds = preds.flatten()
            y = y.float()

        # keep all predictions & groundtruths
        self.output_preds.append(preds)
        self.output_gts.append(y)

        # return loss & acc for this batch
        loss = self.loss_fn(preds, y)
        acc = self.acc(preds, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=True, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx):
        return self._step(batch)

    ####################################################
    # Config data loader for this task (transforms)
    ####################################################
    def _dataloader(self, sub_dir, shuffle=False):
        # self.dataset_root = os.path.abspath(os.path.join(self.dataset_root, os.pardir))
        data_path = os.path.join(self.dataset_root, sub_dir)

        ## setup transform_val
        bCropInput = self.crop_input
        bGrayscale = self.gray_input

        transform_val = transforms.Compose([
            FixedCrop(True) if bCropInput else FixedCrop(False),
            transforms.Grayscale(num_output_channels=3) if bGrayscale else FixedCrop(False),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        ## setup transform_train
        transform_train = transforms.Compose(
            [
                transform_val,
                transforms.RandomHorizontalFlip(0.5),
            ]
        )

        # assign correct transform for train/val/test
        img_folder = ImageFolder(data_path, transform=transform_val)
        if (shuffle):
            img_folder = ImageFolder(data_path, transform=transform_train)

        return DataLoader(img_folder, batch_size=self.batch_size, shuffle=shuffle, num_workers=16, pin_memory=True)

    def train_dataloader(self):
        return self._dataloader("train", shuffle=True)

    def val_dataloader(self):
        return self._dataloader("val")

    def test_dataloader(self):
        return self._dataloader("test")

