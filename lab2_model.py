#%%
import torch
class TinyModel(torch.nn.Module):
    def __init__(self, x=100, w = 2):
        super(TinyModel, self).__init__()
        self.activation = torch.nn.LeakyReLU()
        self.activation2 = torch.nn.Sigmoid()
        self.linear1 = torch.nn.Linear(w, x)
        self.linear2 = torch.nn.Linear(x, x)
        self.linear3 = torch.nn.Linear(x, x)
        self.linear4 = torch.nn.Linear(x, w)
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear4(x)
        x = self.activation2(x)
        return x
    def save(self, patch='lab7_model.pt'):
        torch.save(self.state_dict(), patch)
        print("Model saved: %s" % patch)
    def info(self):
        print(self)
        print("Params: %i" % sum([param.nelement()
                                  for param in self.parameters()]))
if __name__=='__main__':
    w = 2
    tinymodel = TinyModel(w = w)
    print("model params: %i"%sum([param.nelement() for param in tinymodel.parameters()]))
    print('The model:')
    print(tinymodel)
    print('\n\nJust one layer:')
    print(tinymodel.linear2)
    print('\n\nModel params:')
    for param in tinymodel.parameters():
        print(param)
    print('\n\nLayer params:')
    for param in tinymodel.linear2.parameters():
        print(param)
    #test
    x = torch.rand(100, w)
    print("x = ", x.size())
    y = tinymodel(x)
    print("y = ", x.size())
# %%

