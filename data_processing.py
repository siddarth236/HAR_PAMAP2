import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Labels
Lying=0
Idle=1
Walking=2
Running=3
Cycling=4
Ascend=5
Descend=6
Jumping=7

def process_data():
    print("data processing has started")
    input_length=100
    X_cond=[]
    Y_cond=[]
    X_val=[]
    Y_val=[]

    ## Files 10 to 14 are of subject 1,2,6,7,8 doing optional activities. I have renamed the files for convenience during data processing
    files=[1,2,3,4,6,7,8,9,10,11,12,13,14]
    for k in files:
        print(f"Processing file {k}")
        data_1 = pd.DataFrame(np.transpose(np.loadtxt(f'PAMAP2_Dataset\Protocol\subject10{k}.dat', unpack = True)))
        data_1=data_1.drop(data_1.iloc[:,np.r_[2:21,24:27,30:54]],axis=1)
        for j in range(1,25):

            ## selecting all data points of a particular label
            df=(data_1.loc[data_1[1]==j]).reset_index(drop=True)
            ## imputing any nan values with backward fill
            df=df.fillna(method='bfill')

            for i in range(0,len(df),50):
                X=np.array(df.loc[i:i+input_length-1,21:24])
                if(X.shape[0]<input_length):
                    continue
                if(j==1):    
                    X_cond.append(X)
                    Y_cond.append(np.ones(1)*int(Lying))        
                elif(j==2 or j==3 or j==9 or j==10):
                    X_cond.append(X)
                    Y_cond.append(np.ones(1)*int(Idle))   
                elif(j==4 or j==7):
                    X_cond.append(X)
                    Y_cond.append(np.ones(1)*int(Walking))   
                elif(j==5):
                    X_cond.append(X)
                    Y_cond.append(np.ones(1)*Running)   
                elif(j==6):
                    X_cond.append(X)
                    Y_cond.append(np.ones(1)*Cycling)   
                elif(j==12):
                    X_cond.append(X)
                    Y_cond.append(np.ones(1)*Ascend)   
                elif(j==13):
                    X_cond.append(X)
                    Y_cond.append(np.ones(1)*Descend)   
                elif(j==24):
                    X_cond.append(X)
                    Y_cond.append(np.ones(1)*Jumping)   
    X_cond=np.array(X_cond,dtype=np.float32)
    Y_cond=np.array(Y_cond,dtype=np.float32)
    Y_cond=Y_cond.astype(int)
    X_cond=X_cond.reshape(X_cond.shape[0],3,input_length)
    Y_cond=Y_cond.reshape(-1)


    ## Data of subject 5 will only be used for validation and not for training
    files=[5]
    for k in files:
        print(f"Processing file {k}")
        data_1 = pd.DataFrame(np.transpose(np.loadtxt(f'PAMAP2_Dataset\Protocol\subject10{k}.dat', unpack = True)))
        data_1=data_1.drop(data_1.iloc[:,np.r_[2:21,24:27,30:54]],axis=1)
        for j in range(1,25):
            df=(data_1.loc[data_1[1]==j]).reset_index(drop=True)
            df=df.fillna(method='bfill')
            for i in range(0,len(df),100):
                X=np.array(df.loc[i:i+input_length-1,21:24])
                if(X.shape[0]<input_length):
                    continue
                if(j==1):    
                    X_val.append(X)
                    Y_val.append(np.ones(1)*int(Lying))        
                elif(j==2 or j==3 or j==9 or j==10):
                    X_val.append(X)
                    Y_val.append(np.ones(1)*int(Idle))   
                elif(j==4 or j==7):
                    X_val.append(X)
                    Y_val.append(np.ones(1)*int(Walking))   
                elif(j==5):
                    X_val.append(X)
                    Y_val.append(np.ones(1)*Running)   
                elif(j==6):
                    X_val.append(X)
                    Y_val.append(np.ones(1)*Cycling)   
                elif(j==12):
                    X_val.append(X)
                    Y_val.append(np.ones(1)*Ascend)   
                elif(j==13):
                    X_val.append(X)
                    Y_val.append(np.ones(1)*Descend)   
                elif(j==24):
                    X_val.append(X)
                    Y_val.append(np.ones(1)*Jumping)   
    X_val=np.array(X_val,dtype=np.float32)
    Y_val=np.array(Y_val,dtype=np.float32)
    Y_val=Y_val.astype(int)
    X_val=X_val.reshape(X_val.shape[0],3,input_length)
    Y_val=Y_val.reshape(-1)

    #print(type(X_cond[0,0,0]),type(Y_cond[0]))
    print(f'total X data shape: {X_cond.shape} , number of class: {np.unique(Y_cond)} , label shape: {Y_cond.shape}')
    return X_cond,Y_cond,X_val,Y_val

if __name__ =='__main__':
    X,Y,X_val,Y_val=process_data()
