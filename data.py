import torch
import copy
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


class DataPacker:

    def __init__(self,dls):
        self.its=[]
        self.length=0
        for dl in dls:
            self.its.append(iter(dl))
            self.length=max(self.length,len(dl))

    def __iter__(self):
        return self

    def __next__(self):
        ls,batchs=next_items_of_iterators(self.its)
        if len(ls)==0:
            raise StopIteration
        return ls,batchs

    def __len__(self):
        return self.length