from models import MTL_BC
import torch
from collections import namedtuple
import copy
from tqdm import trange
from utils import *
from data import Task


def apply_mask_to_model(model, mask):
    '''
    apply mask to the model
    :param mask:
    :return:
    '''
    for n, p in model.named_parameters():
        if n in mask:
            m=(mask[n] == 0).to(p.data.device)
            p.data.masked_fill_(m, 0.)


def apply_mask_to_grad(model, mask):
    '''
    apply mask to grad of params in model
    :param mask:
    :return:
    '''
    for n, p in model.named_parameters():
        if n in mask:
            m = (mask[n] == 0).to(p.grad.device)
            p.grad.masked_fill_(m, 0.)


def sparsity(mask):
    '''
    calculate the sparsity of a sub-net through mask
    :param mask:
    :return:
    '''
    x,y=0,0
    for name in mask:
        x+=torch.sum(mask[name])
        y+=torch.sum(torch.ones(mask[name].shape))

    return x/y



class Prune:

    def __init__(self,model,tasks,pruning_rate,sparsity,names,warmup_steps,word2id,char2id,path,device):

        '''
        :param model: the model to be pruned
        :param tasks: all the tasks
        :param pruning_rate: the pruning rate in one pruning iteration
        :param sparsity: the min sparsity
        :param names: names of the modules to be pruned in the model
        '''

        self.model=model
        self.tasks=tasks
        self.pr=pruning_rate
        self.min_sparsity=sparsity
        self.names=names
        self.mask_dict={}
        self.warmup_steps=warmup_steps
        self.init_params=None
        self.path=path

        self.word2id=word2id
        self.char2id=char2id
        self.device=device


    def pruning(self):
        '''
        the main function for pruning
        :return:
        '''
        self.init_model()
        self.save_model_params()
        for task in self.tasks:
            masks=[]
            # init the mask with 1
            mask={}
            for name,params in self.model.named_parameters():
                if self.need_prune(name):
                   mask[name]=torch.ones(params.shape).to(self.device)

            while(True):

                self.warmup(task,mask)
                
                # inplace
                self.prune(mask)

                masks.append(copy.deepcopy(mask))

                if sparsity(mask)<self.min_sparsity:
                    break
                else:
                    self.resume_params()
            
            self.mask_dict[task.t_name]=masks

        self.save()



    def save(self):
        '''
        save self.mask_dict to self.path
        :param path:
        :return:
        '''
        torch.save(self.mask_dict,self.path)


    def init_model(self):
        pass

    def save_model_params(self):
        '''
        save model params to self.init_params
        :return:
        '''
        self.init_params=copy.deepcopy(self.model.state_dict())


    def need_prune(self, name):
        '''
        check if the specific module of the model need to be pruned
        :param name:
        :return:
        '''
        for n in self.names:
            if n in name and 'bias' not in name:
                return True
        return False

    def warmup(self, task, mask):
        '''
        warmup the sub-net
        just modify this function when adapt to different model
        :param t_id:
        :param mask:
        :return:
        '''
        optim=torch.optim.Adam(self.model.parameters())
        dl_it=iter(task.train_dl)
        apply_mask_to_model(self.model,mask)
        for step in trange(self.warmup_steps):

            batch=next(dl_it)
            batch = tokenize(batch, task.label2id, self.word2id, self.char2id,self.device)

            loss = self.model.forward_loss(batch['word_ids'], batch['char_ids_f'], batch['word_pos_f']
                                              , batch['char_ids_b'], batch['word_pos_b'], batch['label_ids']
                                              , batch['lens'], task.t_id,need_predict=False)
            loss.backward()
            apply_mask_to_grad(self.model,mask)
            optim.step()
            self.model.zero_grad()

    def prune(self, mask):

        '''
        pruning in sub-net
        :param mask:
        :return:
        '''

        params=torch.tensor([]).to(self.device)
        for n,p in self.model.named_parameters():
            if n in mask:
                params=torch.cat([params,p.data[mask[n]==1]],dim=0)
        params=torch.abs(params)
        params=torch.sort(params).values

        threshold=params[int(self.pr*params.shape[0])]

        for n, p in self.model.named_parameters():
            if n in mask:
                zero=torch.zeros(p.data.shape).to(p.data.device)
                mask[n]=torch.where(torch.abs(p.data)<threshold,zero,mask[n])



    def resume_params(self):
        '''
        load the init params to the model
        :return:
        '''
        self.model.load_state_dict(self.init_params)


if __name__=='__main__':

    print('loading datas')
    dataset_name, datas = load_data('./test_dir')

    vocab = []
    for data in datas:
        vocab += collect_words(list(map(lambda x: x[0], data['train'] + data['devel'] + data['test'])))

    word2id = {w: i for i, w in enumerate(vocab)}

    char = collect_chars(vocab)
    char = ['<pad>'] + char

    char2id = {c: i for i, c in enumerate(char)}

    tasks=[]

    for i in range(len(dataset_name)):
        task=Task(dataset_name[i],i,datas[i]['train'],datas[i]['devel'],datas[i]['test'],1)
        tasks.append(task)

    ds_info = []

    for data in datas:
        ds_info.append(get_dataset_info(data['train'] + data['devel'] + data['test']))

    device=torch.device('cuda')

    model = MTL_BC(len(vocab), len(char), w_emb_size=200, c_emb_size=30, w_hiden_size=300, c_hiden_size=600,
                   dropout_rate=0.5, ds=ds_info, device=device).to(device)

    tem_p=Prune(model,tasks,0.1,0.5,['linear'],1,word2id,char2id,'masks.pt',device)

    tem_p.pruning()


    print('done')






