# init block
# %%
import torch
import os
import matplotlib.pyplot as plt
import sys
import pandas as pd
from comet_ml import Experiment
if 'lab7_model' in sys.modules:
    del sys.modules['lab7_model']
if 'lab7_data' in sys.modules:
    del sys.modules['lab7_data']
from lab7_model import RNNModel as Model
from lab7_data import LoadData

batch = 100
hiden_dim = 8
layer_dim = 3
epoches = 100
steps = 150

experiment = Experiment(project_name="Second Hand Car Data")

dataset_train, dataset_test = LoadData(batch, 10)
x, y = next(iter(dataset_train))
df = pd.read_csv('./archive.zip', low_memory=False)
df = df.rename(
    columns= {'Region': 'region', 'Country': 'country', 'State': 'state', 'City': 'city',
              'Month': 'month', 'Day': 'day', 'Year': 'year', 'AvgTemperature': 'avg_temperature'}
)

    
model = Model(x.size(2), hiden_dim, layer_dim, y.size(1))
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
loss_fn = torch.nn.MSELoss(reduction="mean")
for epoche in range(epoches):
    err = 0.
    for step in range(steps):
        inputs, labels = next(iter(dataset_train))
        
        # normalizacja
        inputs = inputs / torch.Tensor([110.,12.])
        labels = labels / torch.Tensor([110.])

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(inputs)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
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
inputs = inputs / torch.Tensor([110.,12.])
labels = labels / torch.Tensor([110.])
pred = model(inputs).detach()
plt.plot(torch.arange(inputs.size(1)), inputs[0,:,0],'gs'
             ,torch.arange(outputs.size(1))+inputs.size(1),pred[0,:],'bs',
             torch.arange(outputs.size(1))+inputs.size(1),labels[0,:],'rs')


# %%
import plotly.express as px
map_df = df.query('year < 2020').groupby(['country', 'year'])[['avg_temperature']].mean().reset_index()
fig = px.choropleth(
    map_df, locations= 'country', locationmode= 'country names',
    animation_frame= 'year', color= 'avg_temperature',
    hover_name= 'country', color_continuous_scale= 'RdYlBu_r',
    title= 'Average temperature of countries between 1995 - 2019'
)
fig.update_layout(title= {'x': 0.5})
fig.show()
# %%