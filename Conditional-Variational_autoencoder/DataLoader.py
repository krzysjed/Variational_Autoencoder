import setuptools
import os
import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):

    def __init__(self, txt_path, img_dir, transform=None):
        df = pd.read_csv(txt_path, sep="\s+", skiprows=1)
        self.img_dir = img_dir
        self.txt_path = txt_path
        self.img_names = df.index.values
        self.y = df.values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)

        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]

    def __num_attr__(self):
        return self.y.shape[1]


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=1, num_workers=1):
        super().__init__()

        self.data_dir = r'Data'  # path to folder of images
        self.attr_path = 'list_attr.txt'  # path to file.txt of attributes

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def setup(self, stage=None):
        data = MyDataset(self.attr_path, self.data_dir, self.transform)
        self.train_data, self.valid_data = data, data

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_data,
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):

        return DataLoader(self.train_data,
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def __len__(self):
        return MyDataset(self.attr_path, self.data_dir, self.transform).__num_attr__()
