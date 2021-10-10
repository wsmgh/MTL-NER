import torch
import torch.nn as nn


class MTL_BC(nn.Module):

    def __init__(self,vocab_size,char_size,w_emb_size,c_emb_size,w_hiden_size,c_hiden_size,dropout_rate,ds,device):
        super(MTL_BC,self).__init__()
        self.c_hiden_size=c_hiden_size
        self.ds=ds

        self.w_emb=nn.Embedding(vocab_size,w_emb_size)
        self.c_emb=nn.Embedding(char_size,c_emb_size)

        self.c_lstm_f=nn.LSTM(input_size=c_emb_size,hidden_size=c_hiden_size//2,batch_first=True)
        self.c_lstm_b=nn.LSTM(input_size=c_emb_size,hidden_size=c_hiden_size//2,batch_first=True)

        self.w_lstm=nn.LSTM(input_size=w_emb_size+c_hiden_size,hidden_size=w_hiden_size,batch_first=True)

        self.linear=nn.ModuleList([nn.Linear(w_hiden_size,len(ds[i].label2id)) for i in range(len(ds))])

        self.crf=nn.ModuleList([CRF(list(ds[i].label2id.keys()),device) for i in range(len(ds))])

        self.dropout=nn.Dropout(p=dropout_rate)

    def load_pre_word_vec(self,word_vec):
        '''
        导入预训练词向量
        :param word_vec:
        :return:
        '''
        self.w_emb=nn.Embedding.from_pretrained(word_vec,freeze=False)

    def forward(self,input_word_ids,input_char_ids_f,input_word_pos_f,
                     input_char_ids_b,input_word_pos_b,ds_num):
        '''

        :param input_word_ids: batch_size * seq_len
        :param input_char_ids_f: batch_size * seq_len_char
        :param input_word_pos_f: batch_size * seq_len
        :param input_char_ids_b: batch_size * seq_len_char
        :param input_word_pos_b: batch_size * seq_len
        :param ds_num: 数据所属的数据集编号
        :return:
        '''

        #batch_size * seq_len_char * c_emb_size
        char_embeddings_f=self.c_emb(input_char_ids_f)
        char_embeddings_b=self.c_emb(input_char_ids_b)

        char_embeddings_f=self.dropout(char_embeddings_f)
        char_embeddings_b=self.dropout(char_embeddings_b)


        #batch_size * seq_len_char * c_hiden_size
        char_lstm_out_f,_=self.c_lstm_f(char_embeddings_f)
        char_lstm_out_b,_=self.c_lstm_b(char_embeddings_b)


        sp=input_word_pos_f.shape
        input_word_pos_f=input_word_pos_f.unsqueeze(2).expand(sp[0],sp[1],self.c_hiden_size//2)
        input_word_pos_b=input_word_pos_b.unsqueeze(2).expand(sp[0],sp[1],self.c_hiden_size//2)

        #batch_size * seq_len * c_hiden_size/2
        char_lstm_out_f=torch.gather(char_lstm_out_f,1,input_word_pos_f)
        char_lstm_out_b=torch.gather(char_lstm_out_b,1,input_word_pos_b)

        char_lstm_out_f = self.dropout(char_lstm_out_f)
        char_lstm_out_b = self.dropout(char_lstm_out_b)

        # batch_size * seq_len * w_emb_size
        word_embeddings = self.w_emb(input_word_ids)

        word_embeddings=self.dropout(word_embeddings)

        #batch_size * seq_len * c_hiden_size+w_emb_size
        word_embeddings=torch.cat([char_lstm_out_f,char_lstm_out_b,word_embeddings],2)

        #batch_size * seq_len * w_hiden_size
        word_features,_=self.w_lstm(word_embeddings)

        word_features=self.dropout(word_features)

        #batch_size * seq_len * tag_size_of_ds[i]
        scores=self.linear[ds_num](word_features)

        return scores


    def forward_loss(self,input_word_ids,input_char_ids_f,input_word_pos_f,
                          input_char_ids_b,input_word_pos_b,gold_label,lens,ds_num,need_predict=True):
        '''
        :param lens: bath_size
        其余同forward
        :return:
        '''
        scores=self.forward(input_word_ids,input_char_ids_f,input_word_pos_f,input_char_ids_b,input_word_pos_b,ds_num)

        loss=self.crf[ds_num].calculate_loss(gold_label,scores,lens)

        if need_predict:
            labels, _ = self.crf[ds_num].viterbi_decode(scores, lens)
            return loss,labels

        return loss


    def predict(self,input_word_ids,input_char_ids_f,input_word_pos_f,
                     input_char_ids_b,input_word_pos_b,lens,ds_num):
        scores = self.forward(input_word_ids, input_char_ids_f, input_word_pos_f, input_char_ids_b, input_word_pos_b,ds_num)
        labels, _ = self.crf[ds_num].viterbi_decode(scores, lens)
        return labels





class CRF(nn.Module):

    def __init__(self,tagset,device):
        super(CRF, self).__init__()
        '''
        :param tagset: list of all tags
        '''
        self.device=device
        self.tagset_dic={t:i for i,t in enumerate(tagset+['<start>','<stop>'])}
        self.id2tag={i:t for i,t in enumerate(tagset+['<start>','<stop>'])}
        self.tagset_size=len(self.tagset_dic)
        self.trans=nn.Parameter(torch.randn((self.tagset_size,self.tagset_size)),requires_grad=True)


    def calculate_loss(self,gold_label,scores,lens):
        '''
        :param gold_label: batch_size * seq_len
        :param scores: batch_size * seq_len * tagset_size
        :param lens: batch_size
        :return:
        '''
        loss=self.log_exp_score_of_all_labels(scores,lens)-self.log_exp_score_of_gold_label(gold_label,scores,lens)

        return torch.mean(loss)


    def log_exp_score_of_gold_label(self,gold_label,scores,lens):
        '''
        :param gold_label: batch_size * seq_len
        :param scores: batch_size * seq_len * tagset_size
        :param lens: batch_size
        :return: log_exp_scores : batch_size
        '''
        # add <start> and <stop>
        # batch_size * 1
        start_tag=torch.LongTensor([self.tagset_dic['<start>']]).repeat(gold_label.shape[0],1).to(self.device)
        gold_label=torch.cat([start_tag,gold_label,start_tag],dim=1)
        for i in range(gold_label.shape[0]):
            gold_label[i][lens[i]+1]=self.tagset_dic['<stop>']

        log_exp_scores=torch.FloatTensor(gold_label.shape[0])
        for i in range(gold_label.shape[0]):
            gl=gold_label[i,:lens[i]+2]
            idx=gl[1:lens[i]+1].unsqueeze(1)
            e=torch.sum(torch.gather(scores[i,:lens[i]],1,idx))
            t=torch.sum(self.trans[gl[:lens[i]+1],gl[1:lens[i]+2]])
            log_exp_scores[i]=e+t

        return log_exp_scores



    def log_exp_score_of_all_labels(self,scores,lens):
        '''
        :param scores: batch_size * seq_len * tagset_size
        :param lens: batch_size
        :return: log_exp_score_of_all_labels : batch_size
        '''

        log_exp_scores=torch.FloatTensor(scores.shape[0])

        for i in range(scores.shape[0]):

            pre = scores[i][0]+self.trans[-2,:-2]

            for j in range(lens[i]):
                tem_pre=pre.unsqueeze(0).expand(self.tagset_size-2,self.tagset_size-2)
                e=scores[i][j].unsqueeze(1).expand(self.tagset_size-2,self.tagset_size-2)
                t=torch.transpose(self.trans[:-2,:-2],0,1)
                pre=self.log_sum_exp(tem_pre+e+t)

            pre=(pre+self.trans[:-2,-1]).unsqueeze(0)
            log_exp_scores[i]=self.log_sum_exp(pre)

        return log_exp_scores



    def viterbi_decode(self,scores,lens):
        '''
        :param scores: batch_size * seq_len * tagset_size
        :param lens: batch_size
        :return:
        '''

        tags_of_path=[]
        scores_of_path=[]

        for i in range(scores.shape[0]):
            #tagset_size
            path_scores=scores[i][0]+self.trans[-2,:-2]
            ts=path_scores.shape[0]
            path_record=[]
            for j in range(1,lens[i]):
                #tagset_size * tagset_size
                tem_scores=path_scores.unsqueeze(0).expand(ts,ts)
                t=torch.transpose(self.trans[:-2,:-2],0,1)
                e=scores[i][j].unsqueeze(1).expand(ts,ts)
                tem_scores=tem_scores+t+e

                max_score=torch.max(tem_scores,dim=1)

                path_scores=max_score.values
                path_record.append(max_score.indices)

            path_scores=path_scores+self.trans[:-2,-1]
            max_score=torch.max(path_scores,dim=0)
            scores_of_path.append(max_score.values.item())


            tag=max_score.indices.item()
            path=[self.id2tag[tag]]
            path_record.reverse()
            for record in path_record:
                tag=record[tag].item()
                path.append(self.id2tag[tag])
            path.reverse()
            tags_of_path.append(path)


        return tags_of_path,scores_of_path


    def log_sum_exp(self,x):
        '''
        :param x: m * n
        :return: log_sum_exp of each row : m
        '''
        max_values=torch.max(x,dim=1).values
        max_values_batch=max_values[:,None].repeat(1,x.shape[1])
        return max_values+torch.log(torch.sum(torch.exp(x-max_values_batch),dim=1))








if __name__=='__main__':

    from collections import namedtuple

    with open('train.tsv') as f:
        s=[]
        l=[]
        for line in f:
            if line=='\n':
                break
            line=line.strip('\n').split()
            s.append(line[0])
            l.append(line[1])

    vocab=s

    word2id={w:i for i,w in enumerate(vocab)}
    id2word={i:w for i,w in enumerate(vocab)}

    char=[]
    for w in s:
        char+=list(w)
    char=list(set(char))+[' ']

    char2id={c:i for i,c in enumerate(char)}
    id2char={i:c for i,c in enumerate(char)}


    DataSet=namedtuple('DataSet','id2label label2id')

    tagset=['B-Chemical','I-Chemical','B-Disease','I-Disease','O','<START>','<STOP>']
    label2id={w:i for i,w in enumerate(tagset)}
    id2label={i:w for i,w in enumerate(tagset)}

    ds=[DataSet(id2label,label2id)]

    sentence_ids=[word2id[w] for w in s]
    tags=[label2id[t] for t in l]

    ts=' '.join(s)
    char_seq=list(ts+' ')

    def get_space_pos(s):
        pos=[]
        for i,w in enumerate(s):
            if w==' ':
                pos.append(i)
        return pos

    char_ids_f=[char2id[c] for c in char_seq]
    pos_f=get_space_pos(char_seq)



    char_seq.reverse()
    char_seq.append(' ')
    char_seq=char_seq[1:]
    char_ids_b=[char2id[c] for c in char_seq]
    pos_b=get_space_pos(char_seq)
    pos_b.reverse()

    device=torch.device('cpu')

    model=MTL_BC(len(vocab),16,16,16,16,ds,device)


    optim=torch.optim.Adam(model.parameters())
    model.train()

    train_loss=[]

    for i in range(300):

        input_word_ids=torch.tensor([sentence_ids])
        labels=torch.tensor([tags])
        lenghts=torch.tensor([len(sentence_ids)])
        input_char_ids_f=torch.tensor([char_ids_f])
        input_char_ids_b=torch.tensor([char_ids_b])
        input_word_pos_f=torch.tensor([pos_f])
        input_word_pos_b=torch.tensor([pos_b])

        loss=model.forward_loss(input_word_ids,input_char_ids_f,input_word_pos_f,input_char_ids_b,input_word_pos_b,labels,lenghts,0)
        #print(loss)
        train_loss.append(loss.item())

        optim.zero_grad()
        loss.backward()
        optim.step()



    import matplotlib.pyplot as plt

    plt.plot(range(len(train_loss)),train_loss)
    plt.show()

    print('done')

    model.predict(input_word_ids, input_char_ids_f, input_word_pos_f, input_char_ids_b, input_word_pos_b, lenghts, 0)





