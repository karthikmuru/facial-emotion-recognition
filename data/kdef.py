import torch
from torchvision import datasets, models, transforms
import argparse

from .base_data_module  import BaseDataModule

class KDEF(BaseDataModule):

  def __init__(self, args: argparse.Namespace):
    super().__init__(args)

    self.transform = transforms.Compose([transforms.ToTensor()])
    self.mapping =  { 'Afraid': 0,
                      'Angry': 1,
                      'Disgusted': 2,
                      'Happy': 3,
                      'Neutral': 4,
                      'Sad': 5,
                      'Surprised': 6  }

  def setup(self, stage = None):
    data = datasets.ImageFolder(self.data_dir, transform=self.transform)
    test_set_size = int( len(data) * self.split_percentage / 100 )

    self.data_train, self.data_val = torch.utils.data.random_split(data, [test_set_size, len(data) - test_set_size])

  def __repr__(self):
    
    basic = f"KDEF Dataset\nNum classes: {len(self.mapping)}\nMapping: {self.mapping}\n"
    if self.data_train is None and self.data_val is None:
      return basic
    
    x, y = next(iter(self.train_dataloader()))
    data = (
        f"Train/val sizes: {len(self.data_train)}, {len(self.data_val)}\n"
        f"Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
        f"Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
    )

    return basic + data