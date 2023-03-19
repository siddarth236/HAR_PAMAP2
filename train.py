import os
from data_processing import process_data
import matplotlib.pyplot as plt
import numpy as np
import time
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from model import CNN
from dataset import CNN_IMU
from sklearn.metrics import classification_report, confusion_matrix
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_length=100
input_size=3
hidden_feature_size=56
num_classes=8
learning_rate=0.0003527725551849858
batch_size=43
num_epochs=20

X,y,X_val,Y_val=process_data()
X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1,input_size,input_length), y, test_size=0.25, random_state=42)


X_train=np.nan_to_num(X_train)
X_test=np.nan_to_num(X_test)



test_loss = []
train_loss = []
test_accuracy = []
train_accuracy = []
store_loss_train = []
store_loss_test = []

train_dataset=CNN_IMU(X_train,y_train)
first_data=train_dataset[0]
features,labels=first_data


train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_dataset=CNN_IMU(X_test,y_test)

test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)


model=CNN(input_size,hidden_feature_size,num_classes)
model.to(device)
criterion=nn.CrossEntropyLoss()

## Tuned optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,betas=(0.0216,0.9893),weight_decay=1e-6)  

## Default optimizer
#optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

best_accuracy_test = 0
best_accuracy_train = 0

# setting the best loss
best_loss_test = 0
best_loss_train = 0

# Train the model
n_total_steps = len(train_loader)

print(f'Number of Iterations in one epoch:{n_total_steps}')
for epoch in range(num_epochs):
    previous_time=time.time()
    model.train()
    for i, (images, labels) in enumerate(train_loader):  

        images = images.reshape(-1, input_size,input_length).to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)


        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
#         if (i+1) % 10 == 0:
    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f},time_taken:{time.time()-previous_time:.4f}')

    # Test the model
    # In test phase, we don't need to compute gradients 
    with torch.no_grad():
        model.eval()
        train_loss = []
        test_loss = []
        n_correct = 0
        n_samples = 0

        for images, labels in test_loader:
            images = images.reshape(-1, input_size,input_length).to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            test_loss.append(loss.item())

        acc = 100.0 * n_correct / n_samples
        acc=round(acc,4)
        print(f'Accuracy of the network on the test data: {acc} %, loss {np.mean(test_loss):.4f}')
        test_accuracy.append(acc)
        store_loss_test.append(np.mean(test_loss))
        for images, labels in train_loader:
            images = images.reshape(-1, input_size,input_length).to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            train_loss.append(loss.item())
        acc = 100.0 * n_correct / n_samples
        acc=round(acc,4)
        print(f'Accuracy of the network on the train data: {acc} %, loss {np.mean(train_loss):.4f}')
        train_accuracy.append(acc)
        store_loss_train.append(np.mean(train_loss))

        ## Storing the best model
        if ((test_accuracy[-1]) > best_accuracy_test):
            # dir = 'models_save'
            # for f in os.listdir(dir):
            #     os.remove(os.path.join(dir, f))
            file_name = f'group_2_model_best_.h5' 
            print(file_name) 
            best_accuracy_test = np.max(test_accuracy)
            torch.save(model.state_dict(), file_name)

        print(train_accuracy)
        print(test_accuracy)

        ## Storing test and train accuracy and loss
        # a_file = open("test.txt", "w")
        # np.savetxt(a_file, np.array(test_accuracy))
        # a_file.close()

        # a_file = open("train.txt", "w")
        # np.savetxt(a_file, np.array(train_accuracy))
        # a_file.close()

        # a_file = open("test_loss.txt", "w")
        # np.savetxt(a_file, np.array(store_loss_test))
        # a_file.close()

        # a_file = open("train_loss.txt", "w")
        # np.savetxt(a_file, np.array(store_loss_train))
        # a_file.close()


## Using best model to validate on validation dataset


MODEL_PATH = 'group_2_model_best_.h5'
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()
X_val=torch.from_numpy(X_val).to(device)
X_val=X_val.type(torch.cuda.FloatTensor)
pred=[]
for i in range(0,X_val.shape[0],80):
    Y_predict=model.forward_run(X_val[i:i+80,:,:])
    _, predicted = torch.max(Y_predict, 1)
    predicted=predicted.to('cpu')
    predicted=predicted.detach().numpy()
    pred.extend(predicted)
print(classification_report(Y_val[:len(pred)],pred))



## Using best model for confusion matrix with test dataset

# MODEL_PATH = 'group_2_model_best_.h5'

# model.load_state_dict(torch.load(MODEL_PATH))
# model.to(device)
# model.eval()
# pred=[]
# X_v=torch.from_numpy(X_test).to(device)
# X_v=X_v.type(torch.cuda.FloatTensor)
# for i in range(0,X_v.shape[0],80):
#     Y_predict=model.forward_run(X_v[i:i+80,:,:])
#     _, predicted = torch.max(Y_predict, 1)
#     predicted=predicted.to('cpu')
#     predicted=predicted.detach().numpy()
#     pred.extend(predicted)
# from sklearn.metrics import confusion_matrix
# import seaborn as sns; sns.set()
# import matplotlib.pyplot as plt
# mat = confusion_matrix(y_test, pred)
# sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,)
# plt.xlabel('true label')
# plt.ylabel('predicted label')