import torch

from data import *
from utils import *
from tqdm import tqdm
from models import MTL_BC
from metrics import *
import copy

def train():

    print('loading datas')
    dataset_name,datas=load_data('./data')

    vocab = []
    for data in datas:
        vocab += collect_words(list(map(lambda x: x[0], data['train']+data['devel']+data['test'])),unique=False)

    print('filter rare words')
    #去掉低频词(词频小于5)
    vocab=rare_word_filter(vocab,5)
    #vocab=['<pad>']+vocab+['<unk>'] 

    word2id = {w: i for i, w in enumerate(vocab)}

    char = collect_chars(vocab)
    char = ['<pad>'] + char

    char2id = {c: i for i, c in enumerate(char)}
    
    print('parsing dataset')
    # 获得不同数据集的标签
    ds_info = []

    for data in datas:
        ds_info.append(get_dataset_info(data['train']+data['devel']+data['test']))

    # 创建dataset对象和dataloader对象
    ds, dl = [], []

    for data in datas:
        tem_ds={}
        tem_dl={}
        for i in data.keys():
            tem_ds[i]=NerDataset(data[i])
            tem_dl[i]=DataLoader(tem_ds[i], batch_size=32)

        ds.append(tem_ds)
        dl.append(tem_dl)
    
    print('load pre-train word vec')
    word2id,pre_word_emb,_=load_embedding_wlm('./word_vec/wikipedia-pubmed-and-PMC-w2v.bin',b' ',word2id,word2id.keys(),True,'<unk>','<pad>',200)

    device = torch.device('cuda')

    print('building model')
    model = MTL_BC(len(vocab), len(char),w_emb_size=200,c_emb_size=30,w_hiden_size=300,c_hiden_size=600,dropout_rate=0,ds=ds_info,device=device).to(device)

    pre_word_emb=pre_word_emb.to(device)

    model.load_pre_word_vec(pre_word_emb)

    #optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.05)
    optim = torch.optim.Adam(model.parameters())

    tem={}
    for i in range(len(dataset_name)):
        tem[i]=[]

    train_loss=copy.deepcopy(tem)
    devel_loss=copy.deepcopy(tem)
    train_acc=copy.deepcopy(tem)
    devel_acc=copy.deepcopy(tem)
    train_f1=copy.deepcopy(tem)
    devel_f1=copy.deepcopy(tem)
    
    print('starting training loop')
    tot_epoch=100
    for epoch in range(tot_epoch):

        # 初始化各个数据集的信息
        it_train,it_devel,it_test=[],[],[]
        for d in dl:
            for k in d.keys():
                if k=='train':
                    it_train.append(d[k])
                elif k=='devel':
                    it_devel.append(d[k])
                elif k=='test':
                    it_test.append(d[k])


        dpacker=DataPacker(it_train)

        model.train()
        loss_batch, acc_batch, f1_batch = copy.deepcopy(tem), copy.deepcopy(tem), copy.deepcopy(tem)

        with tqdm(dpacker,desc='epoch %d/%d training'%(epoch,tot_epoch)) as pbar:
            for mini_batch,i in pbar:

                # 训练一个大batch（由来自各个数据集的小batch组成）

                batch = tokenize(mini_batch, ds_info[i].label2id, word2id, char2id, device)

                loss,labels = model.forward_loss(batch['word_ids'], batch['char_ids_f'], batch['word_pos_f']
                                              , batch['char_ids_b'], batch['word_pos_b'], batch['label_ids']
                                              , batch['lens'], i)

                optim.zero_grad()
                loss.backward()
                optim.step()

                labels=get_tag_id(labels,ds_info[i].label2id)

                acc_batch[i].append(acc(batch['label_ids'],labels,batch['lens']))

                f1s=f_score(batch['label_ids'],labels,len(ds_info[i].label2id)-3,batch['lens'])
                f1_batch[i].append(torch.mean(f1s).item())

                loss_batch[i].append(loss.item())

                pbar.set_postfix_str(dataset_name[i]+'-f1='+f1_batch[i][-1])


                

        for k in loss_batch.keys():
            train_loss[k].append(torch.mean(torch.tensor(loss_batch[k])).item())
            train_acc[k].append(torch.mean(torch.tensor(acc_batch[k])).item())
            train_f1[k].append(torch.mean(torch.tensor(f1_batch[k])).item())

        print('dataset_name  train_f1  train_acc  train_loss')
        for k in train_loss:
            print(dataset_name[k],' ',train_f1[k][-1], ' ', train_acc[k][-1], ' ', train_loss[k][-1])


        model.eval()
        dpacker = DataPacker(it_devel)
        loss_batch, acc_batch, f1_batch = copy.deepcopy(tem), copy.deepcopy(tem), copy.deepcopy(tem)

        with tqdm(dpacker,desc='epoch %d/%d validating'%(epoch,tot_epoch)) as pbar:
            for mini_batch,i in pbar:


                batch = tokenize(mini_batch, ds_info[i].label2id, word2id, char2id, device)

                loss,labels = model.forward_loss(batch['word_ids'], batch['char_ids_f'], batch['word_pos_f']
                                          , batch['char_ids_b'], batch['word_pos_b'], batch['label_ids']
                                          , batch['lens'], i)

                labels = get_tag_id(labels, ds_info[i].label2id)

                acc_batch[i].append(acc(batch['label_ids'], labels, batch['lens']))

                f1s = f_score(batch['label_ids'], labels, len(ds_info[i].label2id) - 3, batch['lens'])
                f1_batch[i].append(torch.mean(f1s).item())

                loss_batch[i].append(loss.item())
                



        for k in loss_batch.keys():
            devel_loss[k].append(torch.mean(torch.tensor(loss_batch[k])).item())
            devel_acc[k].append(torch.mean(torch.tensor(acc_batch[k])).item())
            devel_f1[k].append(torch.mean(torch.tensor(f1_batch[k])).item())

        print('dataset_name  devel_f1  devel_acc  devel_loss')
        for k in devel_loss:
            print(dataset_name[k], ' ', devel_f1[k][-1], ' ', devel_acc[k][-1], ' ', devel_loss[k][-1])


    torch.save(model.state_dict(),'best-model.pt')
    print('done')
    




if __name__ == '__main__':


    train()

    '''

    #pre-process 获得vocab、char等信息

    files=['./test.txt']

    datas = []

    for file in files:
        datas.append(read_data(file))

    vocab = []
    for data in datas:
        vocab+=collect_words(list(map(lambda x:x[0],data)))
    vocab = ['<pad>'] + list(set(vocab))

    word2id = {w: i for i, w in enumerate(vocab)}
    id2word = {i: w for i, w in enumerate(vocab)}

    char = collect_chars(vocab)
    char = ['<pad>'] + char

    char2id = {c: i for i, c in enumerate(char)}
    id2char = {i: c for i, c in enumerate(char)}



    #获得不同数据集的标签
    ds_info=[]

    for data in datas:
        ds_info.append(get_dataset_info(data))

    #创建dataset对象和dataloader对象
    ds,dl=[],[]

    for data in datas:
        ds.append(NerDataset(data))
        dl.append(DataLoader(ds[-1], batch_size=2))


    device = torch.device('cpu')

    from models import MTL_BC

    model = MTL_BC(len(vocab), len(char), 16, 16, 16, 16, ds_info, device).to(device)

    optim = torch.optim.Adam(model.parameters())

    train_loss = []

    #训练
    for epoch in range(2):

        #初始化各个数据集的信息
        it=[iter(d) for d in dl]
        ls,batchs=next_items_of_iterators(it)

        while True:

            #训练一个大batch（由来自各个数据集的小batch组成）
            for i in ls:
                batch = tokenize(batchs[i], ds_info[i].label2id, word2id, char2id, device)

                loss = model.forward_loss(batch['word_ids'], batch['char_ids_f'], batch['word_pos_f']
                                          , batch['char_ids_b'], batch['word_pos_b'], batch['label_ids']
                                          , batch['lens'], i)

                optim.zero_grad()
                loss.backward()
                optim.step()

                train_loss.append(loss.item())

            #获取下一批数据
            ls,batchs=next_items_of_iterators(it)
            if len(ls)==0:
                break



    print('done')
    
    '''
