import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class LogDataset(Dataset):
    """
    Custom PyTorch Dataset for handling log sequences, labels, indicators, and session IDs.
    Ensures all returned values are PyTorch tensors.
    """

    def __init__(self, sequence, label, indicator, session):
        self.sequence = sequence
        self.label = label
        self.indicator = indicator
        self.session = session

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (self.sequence[idx], self.label[idx], self.indicator[idx], self.session[idx])


class LogDataLoader:
    """
    Handles the creation of DataLoaders from datasets.
    """

    def __init__(self, batch_size_train, batch_size_test):
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test

    def create_dataloader(self, data, batch_size):
        sequence = np.array(data['Encoded'].tolist())
        label = data['Label'].tolist()
        indicator = np.array(data['Indicator'].tolist())
        session = data['Session'].tolist()

        dataset = LogDataset(sequence, label, indicator, session)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    def create_dataloaders(self, train_data, test_normal_clean, test_abnormal_clean, test_normal):
        train_loader = self.create_dataloader(train_data, self.batch_size_train)
        test_normal_loader_clean = self.create_dataloader(test_normal_clean, self.batch_size_test)
        test_abnormal_loader_clean = self.create_dataloader(test_abnormal_clean, self.batch_size_test)
        test_normal_loader = self.create_dataloader(test_normal, self.batch_size_test)

        return train_loader, test_normal_loader_clean, test_abnormal_loader_clean, test_normal_loader
