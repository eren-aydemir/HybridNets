import torch
from torchvision import transforms
from utils.utils import DataLoaderX

from pytorch_lightning import LightningDataModule

from hybridnets.dataset import BddDataset

class HybridNetsDataModule(LightningDataModule):
    def __init__(self, params, opt, seg_mode):
        super().__init__()
        self.params = params
        self.opt = opt
        self.seg_mode = seg_mode

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_set = BddDataset(
            params=self.params,
            is_train=True,
            inputsize=self.params.model['image_size'],
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.params.mean, std=self.params.std
                )
            ]),
            seg_mode=self.seg_mode,
            debug=self.opt.debug
            )

            self.valid_dataset = BddDataset(
                params=self.params,
                is_train=False,
                inputsize=self.params.model['image_size'],
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=self.params.mean, std=self.params.std
                    )
                ]),
                seg_mode=self.seg_mode,
                debug=self.opt.debug
            )

    def train_dataloader(self):
        return DataLoaderX(
            self.train_set,
            batch_size=self.opt.batch_size,
            shuffle=True,
            num_workers=self.opt.num_workers,
            pin_memory=self.params.pin_memory,
            collate_fn=BddDataset.collate_fn
        )

    def val_dataloader(self):
        return DataLoaderX(
            self.val_set,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=self.opt.num_workers,
            pin_memory=self.params.pin_memory,
            collate_fn=BddDataset.collate_fn
        )

    #def test_dataloader(self):
    #    return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)