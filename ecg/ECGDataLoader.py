import torch
from torch.utils.data import Dataset, DataLoader, random_split
import lightning as L
import torchvision.transforms as transforms  # Optional, depends on your data format
from Transform import random_shift_1d, ComposeTransforms
from ECGDataset import ECGDataset
class ECGDataLoader(L.LightningDataModule):

    def __init__(self, csv_file, data_dir,fold_train,fold_test, batch_size, num_workers, split_ratio=0.9):
        super().__init__()
        self.csv_file = csv_file
        self.data_dir = data_dir
        self.fold_train = fold_train
        self.fold_test = fold_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = split_ratio

    def prepare_data(self):
        """
        Use this method to download or preprocess data if required.
        Not called during distributed training.
        """
        pass

    def setup(self, stage=None):
        """
        Create train, validation, and test datasets.
        Called on every GPU in distributed training.
        """

        dataset = ECGDataset(
            csv_file=self.csv_file,
            data_dir=self.data_dir,
            fold_list=self.fold_train
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
            fold_list=self.fold_test
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
