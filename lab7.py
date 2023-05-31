# init block
# %%
import torch
import os
import matplotlib.pyplot as plt
import sys
import pandas as pd
#from comet_ml import Experiment
if 'lab7_model' in sys.modules:
    del sys.modules['lab7_model']
if 'lab7_data' in sys.modules:
    del sys.modules['lab7_data']
from lab2_model import TinyModel as Model
from lab7_data import LoadData

batch = 100
hiden_dim = 8
layer_dim = 3
epochs = 30
steps = 150

#experiment = Experiment(project_name="Second Hand Car Data")

dataset_train, dataset_test = LoadData(batch, batch)
x, y = next(iter(dataset_train))
#df1 = pd.read_csv('car_data1.csv', low_memory=False)
#df2 = pd.read_csv('car_data2.csv', low_memory=False)

    
model = Model()
model.info()


# train block
# %%
# model.load()
# cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: %s" % device)
model = model.to(device)

print("Train")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# https://pytorch.org/docs/stable/nn.html#loss-functions
loss_fn = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
for epoche in range(epochs):
    err = 0.
    for step in range(steps):
        inputs, labels = next(iter(dataset_train))
        
        
        # data is already normalized
        #inputs = inputs / torch.Tensor([110.,12.])
        #labels = labels / torch.Tensor([110.])

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(inputs)
        # Compute the loss and its gradients
        loss = loss_fn(outputs[:, 0], labels)
        loss.backward()
        # Model parameters optimization
        optimizer.step()
        
        err += loss.item()
        print("\rerror = %f "%(err), end="")        
    print("\repoch= %d error= %f: \n"%(epoche,err/step))        
model.save()

#%%
model.eval()
inputs, labels = next(iter(dataset_test))
inputs = model(inputs).detach()
outputs = model(inputs).detach()
plt.scatter(inputs[:, 0], outputs[:, 0], c='red')
plt.show()
# %%

#3d Visualization
#https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
#https://likegeeks.com/3d-plotting-in-python/
inputs, labels = next(iter(dataset_test))
inputs = model(inputs).detach()
outputs = model(outputs).detach()
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(inputs[:,0], labels, outputs[:,0])
plt.show()
# %%
