from data import *
from utils import *
from tqdm import *

if __name__ == '__main__':

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
