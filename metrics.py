import torch

def statistic(x, y, c,lens):
    '''
    x:真实类别标签
    y:预测类别标签
    x,y中的类别标签从1开始编号，0用于padding
    x,y shape: batch_size * seq_len
    c:类别的数量
    '''
    result = []
    for i in range(1,c+1):

        tp,fp,tn,fn=0,0,0,0

        for j in range(len(x)):

            tx=x[j,:lens[j]]
            ty=y[j,:lens[j]]

            t = torch.ones(lens[j]) * i

            t = t.to(y.device)

            p = (ty == t)
            n = (ty != t)

            tp += torch.sum((ty[p] == tx[p])).item()
            fp += torch.sum((ty[p] != tx[p])).item()
            tn += torch.sum((ty[n] == tx[n])).item()
            fn += torch.sum((ty[n] != tx[n])).item()

        result.append([tp, fp, tn, fn])
    # result: c*4
    return torch.FloatTensor(result)


def f_score(x, y, c,lens, b=1,epsilon=1e-10):
    # b=1 时，为 f1
    '''
        x:真实类别标签
        y:预测类别标签
        x,y中的类别标签从1开始编号，0用于padding
        x,y shape: 只要x,y的形状一样即可
        c:类别的数量
        epsilon:防止除零而设的小常数
    '''

    # c * 4
    s=statistic(x,y,c,lens)

    result=[]
    for i in range(s.shape[0]):
        recall=(s[i][0]/(s[i][0]+s[i][3]+epsilon)).item()
        precision=(s[i][0]/(s[i][0]+s[i][1]+epsilon)).item()
        f1=((1+b*b)*recall*precision)/(b*b*precision+recall+epsilon)
        result.append(f1)

    #result: shape:(c)
    return torch.FloatTensor(result)


def acc(x,y,lens=[],ndigit=6):
    '''
    x:真实类别标签
    y:预测类别标签
    x,y shape: batch_size * seq_len
    '''

    ct,tot=0,0
    for i in range(len(x)):
        ct+=torch.sum(x[i][:lens[i]]==y[i][:lens[i]])
        tot+=lens[i]
    ans=ct/tot
    return round(ans.item(),ndigit)



if __name__=='__main__':
    x=torch.tensor([[1,2,3,4,5]])
    y=torch.tensor([[1,2,3,4,5]])
    print(f_score(x,y,5,[5]))
    print(acc(x,y,[5]))
