import torch
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset
from collections import namedtuple
import random
from utils import collect_words, next_items_of_iterators, my_next, load_data, read_data
import os


class Task:

    def __init__(self,t_name,t_id,train_data,devel_data,test_data,batch_size):
        self.t_name=t_name
        self.t_id=t_id
        self.train_dl=DataLoader(NerDataset(train_data),batch_size=batch_size)
        self.devel_dl=DataLoader(NerDataset(devel_data),batch_size=batch_size)
        self.test_dl=DataLoader(NerDataset(test_data),batch_size=batch_size)
        self.batch_size=batch_size
        tagset = collect_words(list(map(lambda x: x[1], train_data+devel_data+test_data)))
        self.label2id = {w: i for i, w in enumerate(tagset)}


    def get_train_dataloader(self):
        return self.train_dl

    def get_devel_dataloader(self):
        return self.devel_dl

    def get_test_dataloader(self):
        return self.test_dl



class NerDataset(Dataset):

    def __init__(self,data=[]):
        self.data=data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i][0],self.data[i][1]

DataSetInfo = namedtuple('DataSet', 'id2label label2id')


class DataPacker:

    def __init__(self,dls,keep_order=False):
        self.keep_order=keep_order
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

        if self.keep_order:
            id=self.ls[0]
            self.ls.remove(id)
            self.ls.append(id)
        else:
            random.shuffle(self.ls)
            id=self.ls[0]

        batch=self.batchs[id]
        tem=my_next(self.its[id])
        if tem is None:
            self.ls.remove(id)
        else :
            self.batchs[id]=tem

        return batch,id


    def __len__(self):
        return self.length



def build_tasks(dir='',batch_size=0):

    tasks=[]
    for id,d in enumerate(os.listdir(dir)):
        path=os.path.join(dir,d)

        tem={}

        for f in os.listdir(path):
            if f=='train.tsv':
                train_data=read_data(os.path.join(path,f))
                tem['train']=train_data
            elif f=='devel.tsv':
                devel_data=read_data(os.path.join(path,f))
                tem['devel']=devel_data
            elif f=='test.tsv':
                test_data=read_data(os.path.join(path,f))
                tem['test']=test_data

        task=Task(d,id,tem['train'],tem['devel'],tem['test'],batch_size)

        tasks.append(task)

    return tasks


if __name__=='__main__':

    tasks = build_tasks('test_dir', 32)
    print('done')