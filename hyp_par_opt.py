import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,SubsetRandomSampler
from data_processing import process_data
from model_validation import CNN
from dataset import CNN_IMU
from sklearn.model_selection import KFold, train_test_split
import optuna
from optuna.samplers import TPESampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model,device,dataloader,loss_fn,optimizer):
    train_loss,train_correct=0.0,0
    model.train()
    for X,Y in dataloader:

        X,Y = X.to(device),Y.to(device)
        optimizer.zero_grad()
        Y_pred = model(X)
        loss = loss_fn(Y_pred,Y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X.size(0)
        _, predictions = torch.max(Y_pred.data, 1)
        train_correct += (predictions == Y).sum().item()

    return train_loss,train_correct

def valid_epoch(model,device,dataloader,loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    for X,Y in dataloader:
        
        X,Y = X.to(device),Y.to(device)
        Y_pred = model(X)
        loss=loss_fn(Y_pred,Y)
        valid_loss+=loss.item()*X.size(0)
        _, predictions = torch.max(Y_pred.data,1)
        val_correct+=(predictions == Y).sum().item()

    return valid_loss,val_correct

def cross_val_acc(X,Y,batch_size=64,input_size=100,hidden_size=50,num_classes=8,
                    beta_1=0.9,beta_2=0.99,num_epochs=30,learning_rate=1e-3,k=5):
    criterion = nn.CrossEntropyLoss()

    ## Using Kfold cross validation with k=5
    splits=KFold(n_splits=k,shuffle=True,random_state=42)
    valid_ac=[]
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(Y)))):

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        dataset=CNN_IMU(X,Y)
        train_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=train_sampler)
        valid_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=test_sampler)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model=CNN(input_size=input_size,hidden_size=hidden_size,
                    num_classes=num_classes,drop_1=0.2,drop_2=0.3).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,betas=(beta_1,beta_2))

        history = {'test_loss': [],'valid_acc':[]}

        for epoch in range(num_epochs):
            train_loss, train_correct=train_epoch(model,device,train_loader,criterion,optimizer)
            valid_loss, test_correct=valid_epoch(model,device,valid_loader,criterion)

            train_loss = train_loss / len(train_loader.sampler)
            valid_loss = valid_loss / len(valid_loader.sampler)
            valid_acc = test_correct / len(valid_loader.sampler) * 100

            history['test_loss'].append(valid_loss)
            history['valid_acc'].append(valid_acc)
        valid_ac.append(np.max(history['valid_acc']))
        
    ## the mean test accuracy of the 5 fold validation is the objective function to be maximized
    val_acc=np.mean(np.array(valid_ac))
    return val_acc


def hypertune_model(X,Y):
    def optimize(trial):

        ## Tuning 6 Hyper parameters  
        params={'learning_rate':trial.suggest_loguniform('learning_rate',1e-6,0.01),
                'batch_size':trial.suggest_int('batch_size',16,100),
                'beta_1':trial.suggest_uniform('beta_1',0.01,0.99),
                'beta_2':trial.suggest_uniform('beta_2',0.01,0.99),
                'hidden_size':trial.suggest_int('hidden_size',20,100)
                }
        accuracy=cross_val_acc(X=X,Y=Y,input_size=3,hidden_size=params['hidden_size'],num_classes=8,beta_1=params['beta_1'],
                        beta_2=params['beta_2'],learning_rate=params['learning_rate'],
                        batch_size=params['batch_size'],)
        return accuracy
    
    ## Using Tree-Structured Parzan Sampler with first 8 iterations as random samples
    study=optuna.create_study(direction='maximize',sampler=TPESampler(n_startup_trials=8,n_ei_candidates=9))
    study.optimize(optimize,n_trials=25)
    return study.best_trial


if __name__=='__main__':

    X,Y,_,_=process_data()
    X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1,3,100), Y, test_size=0.25, random_state=42)
    res = hypertune_model(X_train, y_train)
    print(res)
    openfile=open('open.pickle','wb')
    pickle.dump(res,openfile)
    openfile.close()