import torch
import pandas as pd
from torch.utils.data import Dataset
import os

class ECGDataset(Dataset):
    def __init__(
        self,
        csv_file,
        data_dir,
        fold_list,
        transform=None
    ):
        self.info = pd.read_csv(os.path.join(data_dir, csv_file))
        self.fold_list = fold_list
        if fold_list is not None:
            self.info = self.info[self.info["fold"].isin(fold_list)].reset_index(drop=True)
        # Select only rows where label is "MI" or "Healthy"
        self.info = self.info[self.info["label"].isin(["MI", "Healthy"])].reset_index(drop=True)
        self.data_dir = data_dir
        self.label_dict = {"MI": 0, "Healthy": 1}
        self.transform = transform

    def __len__(self):
        return len(self.info)
    
    def get_fold_list(self):
        return self.fold_list

    def get_patient_list(self):
        return self.info["patient_number"].drop_duplicates(inplace=False).tolist()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        heartbeat = torch.load(os.path.join(self.data_dir, self.info.iloc[idx, 1]), weights_only=True)
        label = self.label_dict[self.info.iloc[idx, 2]]
        patient_number = self.info.iloc[idx, 0]
        fold = self.info.iloc[idx, 3]
        if self.transform:
            heartbeat = self.transform(heartbeat)

        return heartbeat, label, patient_number
