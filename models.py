import torch
import torch.nn as nn
from crf import CRF


class MTL_BC(nn.Module):

    def __init__(self,vocab_size,char_size,w_emb_size,c_emb_size,w_hiden_size,c_hiden_size,ds,device):
        super(MTL_BC,self).__init__()
        self.c_hiden_size=c_hiden_size
        self.ds=ds

        self.w_emb=nn.Embedding(vocab_size,w_emb_size)
        self.c_emb=nn.Embedding(char_size,c_emb_size)

        self.c_lstm_f=nn.LSTM(input_size=c_emb_size,hidden_size=c_hiden_size//2,batch_first=True)
        self.c_lstm_b=nn.LSTM(input_size=c_emb_size,hidden_size=c_hiden_size//2,batch_first=True)

        self.w_lstm=nn.LSTM(input_size=w_emb_size+c_hiden_size,hidden_size=w_hiden_size,batch_first=True)

        self.linear=nn.ModuleList([nn.Linear(w_hiden_size,len(ds[i].label2id)) for i in range(len(ds))])

        self.crf=nn.ModuleList([CRF(len(ds[i].label2id),ds[i].label2id,device) for i in range(len(ds))])

    def load_pre_word_vec(self,word_vec):
        '''
        导入预训练词向量
        :param word_vec:
        :return:
        '''
        self.w_emb=nn.Parameter(word_vec)

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

        char_embeddings_f=self.c_emb(input_char_ids_f)
        char_embeddings_b=self.c_emb(input_char_ids_b)

        #batch_size * seq_len_char * c_hiden_size
        char_lstm_out_f,_=self.c_lstm_f(char_embeddings_f)
        char_lstm_out_b,_=self.c_lstm_b(char_embeddings_b)

        sp=input_word_pos_f.shape
        input_word_pos_f=input_word_pos_f.unsqueeze(2).expand(sp[0],sp[1],self.c_hiden_size//2)
        input_word_pos_b=input_word_pos_b.unsqueeze(2).expand(sp[0],sp[1],self.c_hiden_size//2)

        #batch_size * seq_len * c_hiden_size/2
        char_lstm_out_f=torch.gather(char_lstm_out_f,1,input_word_pos_f)
        char_lstm_out_b=torch.gather(char_lstm_out_b,1,input_word_pos_b)

        # batch_size * seq_len * w_emb_size
        word_embeddings = self.w_emb(input_word_ids)

        #batch_size * seq_len * c_hiden_size+w_emb_size
        word_embeddings=torch.cat([char_lstm_out_f,char_lstm_out_b,word_embeddings],2)

        #batch_size * seq_len * w_hiden_size
        word_features,_=self.w_lstm(word_embeddings)

        #batch_size * seq_len * tag_size_of_ds[i]
        scores=self.linear[ds_num](word_features)

        return scores


    def forward_loss(self,input_word_ids,input_char_ids_f,input_word_pos_f,
                          input_char_ids_b,input_word_pos_b,tags,lens,ds_num):
        '''
        :param lens: bath_size
        其余同forward
        :return:
        '''
        scores=self.forward(input_word_ids,input_char_ids_f,input_word_pos_f,input_char_ids_b,input_word_pos_b,ds_num)

        return self.crf[ds_num].calculate_loss(scores,tags,lens)


    def predict(self,input_word_ids,input_char_ids_f,input_word_pos_f,
                     input_char_ids_b,input_word_pos_b,lens,ds_num):
        scores = self.forward(input_word_ids, input_char_ids_f, input_word_pos_f, input_char_ids_b, input_word_pos_b,ds_num)
        tags,_=self.crf[ds_num]._obtain_labels(scores,self.ds[ds_num].id2label,lens)
        return tags


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




