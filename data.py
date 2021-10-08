import torch
import copy
from utils import *
from torch.utils.data import DataLoader,Dataset
from collections import namedtuple
import random

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
            self.length+=len(dl)
        self.ls, self.batchs = next_items_of_iterators(self.its)

    def __iter__(self):
        return self

    def __next__(self):

        if len(self.ls)==0:
            raise StopIteration

        random.shuffle(self.ls)
        id=self.ls[0]
        batch=self.batchs[id]
        tem=my_next(self.its[id])
        if tem is None:
            self.ls.remove(id)
        else :
            self.batchs[id]=batch

        return batch,id


    def __len__(self):
        return self.length

if __name__=='__main__':
    dataset_name, datas = load_data('./data')

    # 创建dataset对象和dataloader对象
    ds, dl = [], []

    for data in datas:
        tem_ds = {}
        tem_dl = {}
        for i in data.keys():
            tem_ds[i] = NerDataset(data[i])
            tem_dl[i] = DataLoader(tem_ds[i], batch_size=32)

        ds.append(tem_ds)
        dl.append(tem_dl)

    it_train, it_devel, it_test = [], [], []
    for d in dl:
        for k in d.keys():
            if k == 'train':
                it_train.append(d[k])
            elif k == 'devel':
                it_devel.append(d[k])
            elif k == 'test':
                it_test.append(d[k])

    dpacker = DataPacker(it_train)

    for d,i in tqdm(dpacker):
        pass
