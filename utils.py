import torch
from data import *
from collections import namedtuple

def read_data(fpath=''):
    data=[]
    s,l='',''
    with open(fpath) as f:

        for line in f:
            if line=='\n' or line=='':
                data.append((s[1:],l[1:]))
                s,l='',''
                continue

            w,t=line.strip('\n').split()
            s=s+' '+w
            l=l+' '+t

    return data


def tokenize(batch,label2id,word2id,char2id,device):
    seqs, labels = batch
    word_ids=[]
    label_ids=[]
    char_ids_f=[]
    char_ids_b=[]
    word_pos_f=[]
    word_pos_b=[]
    lens=[]
    maxl=-1
    maxl_char=-1
    for i in range(len(seqs)):
        word_ids.append([word2id[w] for w in seqs[i].split()])
        label_ids.append([label2id[t] for t in labels[i].split()])
        length=len(word_ids[-1])
        maxl = max(maxl, length)
        lens.append(length)

        char_seq=list(seqs[i])
        char_seq_f=char_seq+[' ']
        char_seq.reverse()
        char_seq_b=char_seq+[' ']

        maxl_char=max(maxl_char,len(char_seq_f))

        c_ids_f=[char2id[c] for c in char_seq_f]
        c_ids_b=[char2id[c] for c in char_seq_b]

        w_pos_f=get_space_pos(char_seq_f)
        w_pos_b=get_space_pos(char_seq_b)

        char_ids_f.append(c_ids_f)
        char_ids_b.append(c_ids_b)

        word_pos_f.append(w_pos_f)
        word_pos_b.append(w_pos_b)


    for i in range(len(word_ids)):
        num=(maxl-len(word_ids[i]))
        word_ids[i]+=[0]*num
        label_ids[i]+=[0]*num

        word_pos_f[i]+=[0]*num
        word_pos_b[i]+=[0]*num

        num=maxl_char-len(char_ids_f[i])

        char_ids_f[i]+=[0]*num
        char_ids_b[i]+=[0]*num

    word_ids = torch.tensor(word_ids, device=device)
    label_ids = torch.tensor(label_ids, device=device)
    char_ids_f = torch.tensor(char_ids_f, device=device)
    char_ids_b = torch.tensor(char_ids_b, device=device)
    word_pos_f = torch.tensor(word_pos_f, device=device)
    word_pos_b = torch.tensor(word_pos_b, device=device)
    lens=torch.tensor(lens,device=device)




    return {'word_ids':word_ids,'label_ids':label_ids,
            'char_ids_f':char_ids_f,'word_pos_f':word_pos_f,
            'char_ids_b':char_ids_b,'word_pos_b':word_pos_b,
            'lens':lens}


def get_space_pos(s=[]):
    pos=[]
    for i,w in enumerate(s):
        if w==' ':
            pos.append(i)
    return pos

def my_next(it):
    try:
        ans=next(it)
    except StopIteration:
        return None
    return ans

def next_items_of_iterators(iter_list=[]):
    ls = set({})
    items = {}
    for i in range(len(iter_list)):
        items[i] = my_next(iter_list[i])
        if items[i] is not None:
            ls.add(i)
    return ls,items

def collect_words(data=[]):
    vocab=set({})
    for item in data:
        vocab=vocab.union(item.split())
    return list(vocab)

def collect_chars(data=[]):
    chars=set({' '})
    for w in data:
        chars=chars.union(list(w))
    return list(chars)

def get_dataset_info(data=[]):
    DataSetInfo = namedtuple('DataSet', 'id2label label2id')
    tagset=['<pad>']+collect_words(list(map(lambda x:x[1],data)))+['<STOP>','<START>']
    label2id = {w: i for i, w in enumerate(tagset)}
    id2label = {i: w for i, w in enumerate(tagset)}
    dsi=DataSetInfo(id2label,label2id)
    return dsi

