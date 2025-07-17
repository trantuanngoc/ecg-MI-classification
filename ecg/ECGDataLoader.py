import torch
from torch.utils.data import Dataset, DataLoader, random_split
import lightning as L
import torchvision.transforms as transforms  
from Transform import random_shift_1d, ComposeTransforms, roll_1d
from ECGDataset import ECGDataset
class ECGDataLoader(L.LightningDataModule):

    def __init__(self, csv_file, data_dir,fold_train,fold_test, batch_size, num_workers, split_ratio=0.9, sample_before=100, sample_after=100):
        super().__init__()
        self.csv_file = csv_file
        self.data_dir = data_dir
        self.fold_train = fold_train
        self.fold_test = fold_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = split_ratio
        self.sample_before = sample_before
        self.sample_after = sample_after

    def setup(self, stage=None):
        dataset = ECGDataset(
            csv_file=self.csv_file,
            data_dir=self.data_dir,
            fold_list=self.fold_train,
            sample_before=self.sample_before,
            sample_after=self.sample_after
        )

        total_size = len(dataset)
        train_size = int(total_size * self.split_ratio)
        val_size = total_size - train_size

        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )

        self.test_dataset = ECGDataset(
            csv_file=self.csv_file,
            data_dir=self.data_dir,
            fold_list=self.fold_test,
            sample_before=self.sample_before,
            sample_after=self.sample_after
        )
                

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )