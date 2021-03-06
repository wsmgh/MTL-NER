import sys
sys.path.append('../')
from models import CRF
import torch
from torch import nn


class BiLSTM_CRF(nn.Module):

    def __init__(self,vocab_size,emb_size,hidden_size,tasks,device,dropout_rate=0.5):

        super(BiLSTM_CRF, self).__init__()

        self.vocab_size=vocab_size
        self.emb_size=emb_size
        self.hidden_size=hidden_size
        self.tasks=tasks
        self.device=device
        self.dropout_rate=dropout_rate

        self.dropout=nn.Dropout(self.dropout_rate)

        self.embedding=nn.Embedding(self.vocab_size,self.emb_size)

        self.bilstm=nn.LSTM(input_size=self.emb_size,hidden_size=self.hidden_size,batch_first=True,bidirectional=True)

        self.linear=nn.ModuleList([nn.Linear(self.hidden_size*2,len(t.label2id)) for t in self.tasks])

        self.crf=nn.ModuleList([CRF(list(t.label2id),device) for t in self.tasks])

    def forward(self,x,t_id):
        '''
        :param x: batch_size * seq_len
        :return:
        '''

        # batch_size * seq_len * emb_size
        emb=self.embedding(x)

        # batch_size * seq_len * (2*hidden_size)
        bilstm_out,_=self.bilstm(emb)

        bilstm_out=self.dropout(bilstm_out)

        # batch_size * seq_len * tag_size
        score=self.linear[t_id](bilstm_out)

        return score

    def forward_loss(self,x,y,lens,t_id,need_predict=True):
        '''
        :param x: batch_size * seq_len
        :param y: batch_size * seq_len
        :param lens: batch_size
        '''
        score=self.forward(x,t_id)

        loss=self.crf[t_id].calculate_loss(y,score,lens)

        if need_predict:
            labels,_=self.crf[t_id].viterbi_decode(score,lens)
            return loss,labels

        return loss


if __name__=='__main__':
    # from utils import load_data,collect_words,collect_chars,my_next,tokenize
    from utils import *
    from data import *
    from metrics import *
    from prune import *
    from tqdm import tqdm
    import random
    import copy
    print('loading datas')
    dataset_name, datas = load_data('../test_dir')
    print()
    #make sure the two task has the same size in data
    for k in datas[1]:
        length=min(len(datas[0][k]),len(datas[1][k]))
        random.shuffle(datas[0][k])
        random.shuffle(datas[1][k])
        datas[0][k] = datas[0][k][:length]
        datas[1][k]=datas[1][k][:length]

    #build vocab
    vocab = []
    for data in datas:
        vocab += collect_words(list(map(lambda x: x[0], data['train'] + data['devel'] + data['test'])), unique=False)

    char = collect_chars(vocab)
    char2id = {c: i for i, c in enumerate(char)}

    vocab=rare_word_filter(vocab,5)
    word2id = {w: i for i, w in enumerate(vocab)}




    #build task
    tasks=[]
    for i,data in enumerate(datas):
        print('building task:%s ,with task id: %d'%(dataset_name[i],i))
        tasks.append(Task(dataset_name[i],i,data['train'],data['devel'],data['test'],10,shuffle=False))

    #build model
    device=torch.device('cuda')
    model=BiLSTM_CRF(len(vocab),emb_size=100,hidden_size=200,tasks=tasks,device=device).to(device)
    optim=torch.optim.Adam(model.parameters())

    '''
    #load mask and apply to model
    mask=torch.load('mask.pt')
    apply_mask_to_model(model,mask)
    '''


    '''
    #get the params that need to be recorded
    names=['bilstm.weight_ih_l0','bilstm.weight_hh_l0','bilstm.weight_ih_l0_reverse','bilstm.weight_hh_l0_reverse']
    all_params=dict(model.named_parameters())
    pre_params={}
    for name in names:
        pre_params[name]=copy.deepcopy(all_params[name])

    #init increment and decrement
    increment={}
    decrement={}
    for k in pre_params:
        increment[k]=torch.zeros(pre_params[k].shape)
        decrement[k]=torch.zeros(pre_params[k].shape)
    '''

    # start train loop
    f1_epoch = {0: [], 1: []}
    tot_epoch = 10

    for epoch in range(tot_epoch):

        dpacker=DataPacker([tasks[0].train_dl,tasks[1].train_dl],True)
        pbar=tqdm(dpacker,'epoch %d/%d'%(epoch,tot_epoch))

        model.train()
        for batch,t_id in pbar:
            #print(t_id)
            batch=tokenize(batch,tasks[t_id].label2id,word2id,char2id,device)

            loss,label=model.forward_loss(batch['word_ids'],batch['label_ids'],batch['lens'],t_id)

            labels = get_tag_id(label, tasks[t_id].label2id)
            f1s = f_score(batch['label_ids'], labels, len(tasks[t_id].label2id), batch['lens'])
            f1=torch.mean(f1s).item()

            optim.zero_grad()
            loss.backward()

            #apply_mask_to_grad(model,mask)

            optim.step()

            '''
            #get params after backward
            all_params = dict(model.named_parameters())
            cur_params={}
            for name in names:
                cur_params[name] = copy.deepcopy(all_params[name])
                delta=(cur_params[name]-pre_params[name]).to('cpu')
                mask=delta>0
                increment[name][mask]+=delta[mask]
                decrement[name][mask==False] += delta[mask==False]

            pre_params=cur_params

            #param_list.append(copy.deepcopy(dict(model.named_parameters())['bilstm.weight_ih_l0']))
            '''

            pbar.set_postfix_str('%s:%s'%(tasks[t_id].t_name+'-f1',str(f1)))






        dpacker = DataPacker([tasks[0].devel_dl, tasks[1].devel_dl], True)
        pbar = tqdm(dpacker, 'epoch %d/%d' % (epoch, tot_epoch))

        f1_batch = {0: [], 1: []}
        model.eval()
        for batch, t_id in pbar:
            # print(t_id)
            batch = tokenize(batch, tasks[t_id].label2id, word2id, char2id, device)

            loss, label = model.forward_loss(batch['word_ids'], batch['label_ids'], batch['lens'], t_id)

            labels = get_tag_id(label, tasks[t_id].label2id)
            f1s = f_score(batch['label_ids'], labels, len(tasks[t_id].label2id), batch['lens'])
            f1 = torch.mean(f1s).item()
            f1_batch[t_id].append(f1)

            pbar.set_postfix_str('%s:%s' % (tasks[t_id].t_name + '-f1', str(f1)))

        for k in f1_batch:
            f1_epoch[k].append(torch.mean(torch.tensor(f1_batch[k])).item())

        save_result('f1.txt',f1_epoch)

    '''
    r={}
    mask = {}
    for name in names:
        decrement[name]=torch.abs(decrement[name])
        m=increment[name]<decrement[name]
        x=torch.where(m,increment[name],decrement[name])
        y=torch.where(m,decrement[name],increment[name])
        r[name]=x/y
        mask[name]=torch.where(r[name]>0.9,torch.zeros(r[name].shape),torch.ones(r[name].shape))

    torch.save(mask,'mask.pt')
    torch.save(increment,'increment.pt')
    torch.save(decrement,'decrement.pt')
    torch.save(r,'r.pt')
    '''



    #torch.save(param_list,'record_params.pt')
    
    print('done')
