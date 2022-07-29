import torch.nn as nn
import torch
from torchvision.ops.boxes import nms as nms_torch
import torch.nn.functional as F
import math
from functools import partial
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

    def forward(self, imgs, annotations, seg_annot, obj_list=None):
        _, regression, classification, anchors, segmentation = self.model(imgs)

        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
            tversky_loss = self.seg_criterion1(segmentation, seg_annot)
            focal_loss = self.seg_criterion2(segmentation, seg_annot)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
            tversky_loss = self.seg_criterion1(segmentation, seg_annot)
            focal_loss = self.seg_criterion2(segmentation, seg_annot)

        seg_loss = tversky_loss + 1 * focal_loss
        # print("TVERSKY", tversky_loss)
        # print("FOCAL", focal_loss)

        return cls_loss, reg_loss, seg_loss, regression, classification, anchors, segmentation

    def training_step(self, batch, batch_idx):
        imgs, annotations, seg_annot = batch
        cls_loss, reg_loss, seg_loss, _, _, _, _ = self(imgs, annotations, seg_annot)

        return cls_loss, reg_loss, seg_loss


    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), self.opt.lr)

        return [opt], []

