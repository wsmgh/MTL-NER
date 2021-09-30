

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




if __name__=='__main__':

    d=read_data('./test.txt')



    print(d)