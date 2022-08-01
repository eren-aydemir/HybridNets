import torch.nn as nn
import torch
from hybridnets.loss import FocalLoss, FocalLossSeg, TverskyLoss
from hybridnets.model import *

from pytorch_lightning import LightningModule

class ModelWithLightning(LightningModule):
    def __init__(self, model, opt, debug=False):
        super().__init__()
        self.model = model
        self.criterion = FocalLoss()
        self.seg_criterion1 = TverskyLoss(mode=self.model.seg_mode, alpha=0.7, beta=0.3, gamma=4.0 / 3, from_logits=False)
        self.seg_criterion2 = FocalLossSeg(mode=self.model.seg_mode, alpha=0.25)
        self.debug = debug
        self.opt = opt

    def forward(self, imgs):
        _, regression, classification, anchors, segmentation = self.model(imgs)

        return regression, classification, anchors, segmentation

    def losses(self, imgs, annotations, seg_annot):
        regression, classification, anchors, segmentation = self(imgs)

        cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        tversky_loss = self.seg_criterion1(segmentation, seg_annot)
        focal_loss = self.seg_criterion2(segmentation, seg_annot)

        seg_loss = tversky_loss + 1 * focal_loss
        # print("TVERSKY", tversky_loss)
        # print("FOCAL", focal_loss)

        return cls_loss, reg_loss, seg_loss

    def training_step(self, batch, batch_idx):
        imgs = batch['img']
        annotations = batch['annot']
        seg_annot = batch['segmentation']

        cls_loss, reg_loss, seg_loss = self.losses(imgs, annotations, seg_annot)

        cls_loss = cls_loss.mean() if not self.opt.freeze_det else torch.tensor(0)
        reg_loss = reg_loss.mean() if not self.opt.freeze_det else torch.tensor(0)
        seg_loss = seg_loss.mean() if not self.opt.freeze_seg else torch.tensor(0)

        loss = cls_loss + reg_loss + seg_loss

        self.log("cls_loss",    cls_loss,   on_step=True, prog_bar=True)
        self.log("reg_loss",    reg_loss,   on_step=True, prog_bar=True)
        self.log("seg_loss",    seg_loss,   on_step=True, prog_bar=True)
        self.log("total_loss",  loss,       on_step=True, prog_bar=True)

        return loss

    # def validation_step(self, batch, batch_idx):
    #     imgs = batch['img']
    #     annotations = batch['annot']
    #     seg_annot = batch['segmentation']

    #     cls_loss, reg_loss, seg_loss = self.losses(imgs, annotations, seg_annot)

    #     cls_loss = cls_loss.mean() if not self.opt.freeze_det else torch.tensor(0)
    #     reg_loss = reg_loss.mean() if not self.opt.freeze_det else torch.tensor(0)
    #     seg_loss = seg_loss.mean() if not self.opt.freeze_seg else torch.tensor(0)

    #     loss = cls_loss + reg_loss + seg_loss

    #     self.log("val_cls_loss",    cls_loss,   on_step=True, prog_bar=True)
    #     self.log("val_reg_loss",    reg_loss,   on_step=True, prog_bar=True)
    #     self.log("val_seg_loss",    seg_loss,   on_step=True, prog_bar=True)
    #     self.log("val_total_loss",  loss,       on_step=True, prog_bar=True)

    #     return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), self.opt.lr)
        return [opt], []