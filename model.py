import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int,drop_1=0.2,drop_2=0.3):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.hidden_size=hidden_size
        self.num_classes=num_classes
        ks=5
        ps=2

        self.layer_1   = nn.Conv1d(input_size, self.hidden_size, ks, padding=ps)
        self.activation_1 = nn.LeakyReLU(0.02)
        self.pool_1 = nn.MaxPool1d(2)
        self.drop_1 = nn.Dropout(p=drop_1)
        self.layer_2 = nn.Conv1d(self.hidden_size, 128, ks,padding=ps)
        self.drop_2 = nn.Dropout(p=drop_2)
        self.layer_3 = nn.Conv1d(128, 256, ks, padding=ps)
        self.layer_4 = nn.Conv1d(256, 256, ks, padding=ps)
        self.layer_5 = nn.Conv1d(256, 256, ks, padding=ps)
        self.dense_1 = nn.Linear(in_features = 256*3, out_features = 100)
        self.activation_2=nn.ReLU()
        self.dense_2 = nn.Linear(in_features = 100, out_features = self.num_classes)
        self.activation_softmax = nn.Softmax(dim=1)


    def forward(self, x):

        x = self.layer_1(x)
        x = self.activation_1(x)
        x = self.pool_1(x)
        x = self.drop_1(x)
        #print(x.shape)
        

        x = self.layer_2(x)
        x = self.activation_1(x)
        x = self.pool_1(x)
        x = self.drop_2(x)
        #print(x.shape)

        x = self.layer_3(x)
        x = self.activation_1(x)
        x = self.pool_1(x)
        x = self.drop_2(x)
        #print(x.shape)

        x = self.layer_4(x)
        x = self.activation_1(x)
        x = self.pool_1(x)
        x = self.drop_2(x)

        #print(x.shape)
        x = self.layer_5(x)
        x = self.activation_1(x)
        x = self.pool_1(x)
        x = self.drop_2(x)
        #print(x.shape)

        x=self.dense_1(x.reshape(-1,x.shape[1]*x.shape[2]))
        x=self.activation_1(x)
        y=self.dense_2(x)
        #y=x.reshape(-1,self.num_classes)
        #print(y.shape)
        return y

    def forward_run(self,x):
        x=self.forward(x)
        y=self.activation_softmax(x)
        return y

        
if __name__ == '__main__':
    x=torch.randn(10,3,100)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x=x.to(device)
    model=CNN(3,50,8)
    model = model.to(device)
    print( sum(p.numel() for p in model.parameters()))
    print(model(x))


