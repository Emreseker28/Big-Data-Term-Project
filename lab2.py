# init block
#%%
import torch
import sys
from lab2_model import TinyModel as Model
model = Model()
if 'lab7_data' in sys.modules:
    del sys.modules['lab7_data']
from lab7_data import LoadData
batch = 100
epochs = 75
steps = 150
dataset_train, dataset_test = LoadData(batch, 10)
x, y = next(iter(dataset_train))

#train block
#%%
# cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: %s" % device)
model = model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#loss_fn = torch.nn.MSELoss(reduction="mean")
loss_fn = torch.nn.MSELoss(reduction="mean")
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
        loss = loss_fn(outputs[epoche][0], labels)
        loss.backward()
        # Model parameters optimization
        optimizer.step()
        
        err += loss.item()
        print("\rerror = %f "%(err), end="")        
    print("\repoch= %d error= %f: \n"%(epoche,err/step))        
#model.save()
#prediction/test block
#%%
model.eval()
z, y_true = next(iter(dataset_test))
y_pred = model(x)
print(" x=",x)
print("y_pred=",y_pred)
print("y_true=",y_true)
#print("L1_Err=",y_true-y_pred)

#%%
import matplotlib.pyplot as plt
#plt.scatter(x,y)
y_pred = y_pred.detach()
plt.scatter(x, y_pred)
plt.show()

# %%
