import torch
from collections import namedtuple
from collections import Counter
import os
import numpy as np
from tqdm import tqdm

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
    l=len(word2id)-1
    for i in range(len(seqs)):
        word_ids.append([word2id.get(w,l) for w in seqs[i].lower().split()])
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
    ls = []
    items = {}
    for i in range(len(iter_list)):
        items[i] = my_next(iter_list[i])
        if items[i] is not None:
            ls.append(i)
    return ls,items

def rare_word_filter(vocab=[],min_freq=0):

    d=Counter(vocab)
    return list(filter(lambda x:d[x]>=min_freq,d))

def collect_words(data=[],unique=True):
    vocab=[]
    for item in data:
        vocab+=item.split()

    if unique:
        vocab=list(set(vocab))

    return vocab

def collect_chars(data=[]):
    chars=set({' '})
    for w in data:
        chars=chars.union(list(w))
    return list(chars)

def get_dataset_info(data=[]):
    DataSetInfo = namedtuple('DataSet', 'id2label label2id')
    tagset=collect_words(list(map(lambda x:x[1],data)))
    label2id = {w: i for i, w in enumerate(tagset)}
    id2label = {i: w for i, w in enumerate(tagset)}
    dsi=DataSetInfo(id2label,label2id)
    return dsi

def load_data(dir=''):

    name=[]
    data=[]
    for d in os.listdir(dir):
        name.append(d)
        path=os.path.join(dir,name[-1])

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

        data.append(tem)

    return name,data


def get_tag_id(labels=[],label2id={}):
    ids=[]
    maxn=0
    for l in labels:
        ids.append([label2id[t] for t in l])
        maxn=max(maxn,len(l))

    for i in range(len(ids)):
        ids[i]+=[0]*(maxn-len(ids[i]))

    return torch.tensor(ids)



def save_result(file='',data={}):
    with open(file,'w') as f:
        for k in data:
            f.write(' '.join([str(i) for i in data[k]])+'\n')




def load_embedding_wlm(emb_file, delimiter, feature_map, full_feature_set, caseless, unk,pad, emb_len,
                       shrink_to_train=False, shrink_to_corpus=False):
    """
    load embedding, indoc words would be listed before outdoc words

    args:
        emb_file: path to embedding file
        delimiter: delimiter of lines
        feature_map: word dictionary
        full_feature_set: all words in the corpus
        caseless: convert into casesless style
        unk: string for unknown token
        emb_len: dimension of embedding vectors
        shrink: whether to shrink out-of-training set or not
        shrink: whether to shrink out-of-corpus or not
    """
    if caseless:
        feature_set = set([key.lower() for key in feature_map])
        full_feature_set = set([key.lower() for key in full_feature_set])
    else:
        feature_set = set([key for key in feature_map])
        full_feature_set = set([key for key in full_feature_set])

    # ensure <unk> is 0
    word_dict = {v: (k + 1) for (k, v) in enumerate(feature_set - set([unk]))}
    word_dict[unk] = 0

    in_doc_freq_num = len(word_dict)
    rand_embedding_tensor = torch.FloatTensor(in_doc_freq_num, emb_len)
    # init_embedding(rand_embedding_tensor)

    indoc_embedding_array = list()
    indoc_word_array = list()
    outdoc_embedding_array = list()
    outdoc_word_array = list()


    with open(emb_file, 'rb') as f:
        word_count, vec_size = map(int, f.readline().split(delimiter))
        print("word_count: ", word_count, "vec_size: ", vec_size)

        for i in tqdm(range(word_count)):
            word = b''.join(iter(lambda: f.read(1),delimiter))
            word = word.decode('utf-8').lstrip('\n')
            vector = np.fromfile(f, np.float32, vec_size)

            if shrink_to_train and word not in feature_set:
                continue

            if word == unk:
                rand_embedding_tensor[0] = torch.FloatTensor(vector)  # unk is 0
            elif word in word_dict:
                rand_embedding_tensor[word_dict[word]] = torch.FloatTensor(vector)
            elif word in full_feature_set:
                indoc_embedding_array.append(vector)
                indoc_word_array.append(word)
            elif not shrink_to_corpus:
                outdoc_word_array.append(word)
                outdoc_embedding_array.append(vector)




    embedding_tensor_0 = torch.FloatTensor(np.asarray(indoc_embedding_array))

    if not shrink_to_corpus:
        embedding_tensor_1 = torch.FloatTensor(np.asarray(outdoc_embedding_array))
        #word_emb_len = embedding_tensor_0.size(1)
        #assert (word_emb_len == emb_len)

    if shrink_to_corpus:
        embedding_tensor = torch.cat([rand_embedding_tensor, embedding_tensor_0], 0)
    else:
        embedding_tensor = torch.cat([rand_embedding_tensor, embedding_tensor_0, embedding_tensor_1], 0)

    for word in indoc_word_array:
        word_dict[word] = len(word_dict)
    in_doc_num = len(word_dict)
    if not shrink_to_corpus:
        for word in outdoc_word_array:
            word_dict[word] = len(word_dict)


    embedding_tensor=torch.cat([embedding_tensor,embedding_tensor[0].reshape(1,-1)],dim=0)
    word_dict[pad]=0
    word_dict[unk]=embedding_tensor.shape[1]-1

    return word_dict, embedding_tensor, in_doc_num



if __name__=='__main__':

    pass