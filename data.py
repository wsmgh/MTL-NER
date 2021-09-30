import torch

from utils import *
from torch.utils.data import DataLoader,Dataset


class NerDataset(Dataset):

    def __init__(self,data=[]):
        self.data=data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i][0],self.data[i][1]


if __name__=='__main__':
    from collections import namedtuple
    DataSet = namedtuple('DataSet', 'id2label label2id')
    data=read_data('./test.txt')

    vocab=[]
    for item in data:
        vocab+=item[0].split()
    vocab=list(set(vocab))

    word2id={w:i for i,w in enumerate(vocab,1)}
    id2word={i:w for i,w in enumerate(vocab,1)}
    word2id['<pad>'] = 0
    id2word[0] = '<pad>'

    char=set({' '})
    for w in vocab:
        char=char.union(list(w))
    char=list(char)

    char2id={c:i for i,c in enumerate(char,1)}
    id2char={i:c for i,c in enumerate(char,1)}
    char2id['<pad>']=0
    id2char[0]='<pad>'

    tagset=['B-Chemical','I-Chemical','B-Disease','I-Disease','O','<START>','<STOP>']
    label2id={w:i for i,w in enumerate(tagset,1)}
    id2label={i:w for i,w in enumerate(tagset,1)}
    label2id['<pad>'] = 0
    id2label[0] = '<pad>'

    ds_info=[DataSet(id2label,label2id)]

    ds=NerDataset(data)

    dl=DataLoader(ds,batch_size=2)

    device=torch.device('cpu')

    from models import MTL_BC

    model=MTL_BC(len(vocab),len(char),16,16,16,16,ds_info,device).to(device)

    optim=torch.optim.Adam(model.parameters())

    train_loss=[]

    for batch in dl:
        seqs,labels=batch
        batch=tokenize(batch,label2id,word2id,char2id,device)

        loss=model.forward_loss(batch['word_ids'],batch['char_ids_f'],batch['word_pos_f']
                                ,batch['char_ids_b'],batch['word_pos_b'],batch['label_ids']
                                ,batch['lens'],0)

        optim.zero_grad()
        loss.backward()
        optim.step()

        train_loss.append(loss.item())
        print('done')