import torch
from torch.utils.data import Dataset
from data_processing import process_data

class CNN_IMU(Dataset):
    def __init__(self,X, y, size = 100):
        # Initialize data, download, etc.
        # read with numpy or pandas
        
        self.n_samples = X.shape[0]

        # here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(X.reshape(-1,3,size)).float() # size [n_samples, n_features]
        self.y_data = torch.from_numpy(y).long() # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
if __name__ == '__main__':
    X,y=process_data()
    d=CNN_IMU(X,y)
    print((d[1][0].shape))