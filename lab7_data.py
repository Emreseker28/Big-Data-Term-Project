# %%
from typing import Iterator
import torch
import os
import glob
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class DataSet():
    def __init__(self, data, batch=10, x_num=6, y_num=1):
        self.data = data
        self.batch = batch
        self.x_num = x_num
        self.y_num = y_num

    def augment(self, x, y):
        return x, y

    def next_batch(self):
        s = (self.x_num+self.y_num)
        n = self.data.size(0) - s
        idx = (torch.rand(self.batch)*n).long()
        idx = idx.unsqueeze(1).expand(-1, s) + torch.arange(0, s)
        d = self.data[idx]
        
        yield self.augment(d[:, :self.x_num, :], d[:, self.x_num:, 0])

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self.next_batch())

df = None
# data = None

def LoadData(batch_size_train=10, batch_size_test=10):
    global df
    if df is None:
        print("loading datasets")
        df1 = pd.read_csv('car_data1.csv', low_memory=False)
        df2 = pd.read_csv('car_data2.csv', low_memory=False)
    #df1.info()
    # to avoid scientific notation
    #pd.options.display.float_format = '{:.5f}'.format
    #df1.describe()
    
    price1 = df1.iloc[:, 6] #prices from the first dataset
    price2 = df2.iloc[:, 10] #prices from the second dataset
    km1 = df1.iloc[:, 7] #kms from the first dataset
    km2 = df2.iloc[:, 4]
    ca1 = df1.iloc[:, 11] #car age from the first dataset
    ca2 = df2.iloc[:, 3]
    price = pd.concat([price1, price2], axis=0)
    km = pd.concat([km1, km2], axis=0)
    car_age = pd.concat([ca1, ca2], axis=0)
    
    #normalization
    price = (price-price.min())/(price.max()-price.min())
    km = (km-km.min())/(km.max()-km.min())
    car_age = (car_age-car_age.min())/(car_age.max()-car_age.min())
    #print(price)
    #print(km)
    #print(car_age)
    #price.describe()
    #km.describe()
    #car_age.describe()
    data = pd.concat([price.rename('Price'), km.rename('Kilometers'), car_age.rename('Car Age')], axis=1)
    #print(data)
    
    df = data
    
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=0.2, random_state=25)
    #df = pd.DataFrame(data=data)
    #print(df)

    #train = df.sample(frac=0.8, random_state=25)
    #test = df.drop(train.index)
    test_pt = torch.Tensor(test[['Price', 'Kilometers', 'Car Age']].to_numpy())
    train_pt = torch.Tensor(train[['Price', 'Kilometers', 'Car Age']].to_numpy())
    print(f"No. of training examples: {train.shape[0]}")

    # debug
    # self = DataSet(train_pt)

    return DataSet(train_pt, batch_size_train), DataSet(test_pt, batch_size_test)

if __name__ == '__main__':
    train, test = LoadData()
    print(train)
    x, y = next(iter(train)) 
    plt.plot(torch.arange(train.x_num), x[0,:,0],'gs'
             ,torch.arange(train.y_num)+train.x_num,y[0,:],'bs')

# %%
