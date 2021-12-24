import os
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=1, num_workers=1):
        super().__init__()
        self.data_dir = r'Path\Dataset'

        self.batch_size = batch_size
        self.num_workers = num_workers
        # Defining transforms to be applied on the data

        self.transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def prepare_data(self):
        # Downloading our data
        datasets.ImageFolder(os.path.join(self.data_dir, "Train"))

    def setup(self, stage=None):
        # Loading our data after applying the transforms
        data = datasets.ImageFolder(os.path.join(self.data_dir, 'Train'),
                                    self.transform)
        self.train_data, self.valid_data = data, data

    def train_dataloader(self):
        # Generating train_dataloader
        return DataLoader(self.train_data,
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        # Generating val_dataloader
        return DataLoader(self.valid_data,
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        # Generating test_dataloader
        return DataLoader(self.train_data,
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
