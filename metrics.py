import torch

def statistic(x, y, c):
    '''
    x:真实类别标签
    y:预测类别标签
    x,y shape: 只要x,y的形状一样即可
    c:类别的数量
    '''
    result = []
    for i in range(c):
        t = torch.ones(y.shape) * i

        t = t.to(y.device)

        p = (y == t)
        n = (y != t)

        tp = torch.sum((y[p] == x[p])).item()
        fp = torch.sum((y[p] != x[p])).item()
        tn = torch.sum((y[n] == x[n])).item()
        fn = torch.sum((y[n] != x[n])).item()
        result.append([tp, fp, tn, fn])
    # result: c*4
    return torch.FloatTensor(result)


def f_score(x, y, c, b=1,epsilon=1e-10):
    # b=1 时，为 f1
    '''
        x:真实类别标签
        y:预测类别标签
        x,y shape: 只要x,y的形状一样即可
        c:类别的数量
        epsilon:防止除零而设的小常数
    '''

    # c * 4
    s=statistic(x,y,c)

    result=[]
    for i in range(s.shape[0]):
        recall=(s[i][0]/(s[i][0]+s[i][3]+epsilon)).item()
        precision=(s[i][0]/(s[i][0]+s[i][1]+epsilon)).item()
        f1=((1+b*b)*recall*precision)/(b*b*precision+recall+epsilon)
        result.append(f1)

    #result: shape:(c)
    return torch.FloatTensor(result)


def acc(x,y,ndigit=6):
    tx=x.reshape(-1)
    ty=y.reshape(-1)
    ans=torch.sum(tx==ty)/tx.shape[0]
    return round(ans.item(),ndigit)



if __name__=='__main__':
    x=torch.tensor([0,1,2,3,4])
    y=torch.tensor([0,1,2,3,4])
    print(f_score(x,y,5))
    print(acc(x,y))
