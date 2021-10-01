import torch

from utils import *
from torch.utils.data import DataLoader,Dataset
from collections import namedtuple

class NerDataset(Dataset):

    def __init__(self,data=[]):
        self.data=data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i][0],self.data[i][1]

DataSetInfo = namedtuple('DataSet', 'id2label label2id')

