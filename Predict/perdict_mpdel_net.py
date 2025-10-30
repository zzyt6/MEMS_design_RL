import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets  #torchvision/torchtext/torchAudio 都包含数据集  
from torchvision.transforms import ToTensor


train_data = datasets.FashionMNIST(root="../datasets",
train= True,transform= ToTensor(),
download=True)
test_data = datasets.FashionMNIST(root="../datasets",
train= False,transform= ToTensor(),
download=True)
#TOTENSOR将目标值缩放到了我们【0，1】之间


train_dataload = DataLoader(train_data,batch_size=64,shuffle = True,num_workers = 4,drop_last = True)
test_dataload = DataLoader(test_data,batch_size=64,shuffle = True,num_workers = 4,drop_last = True)



from turtle import forward


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print("decive = {}".format(device))

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.liner_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self,x):
        x = self.flatten(x)
        x = self.liner_relu_stack(x)
        return x

predict_model = MLP()
predict_model.to(device)



import torch.optim as optim
loss = nn.CrossEntropyLoss()
optims = optim.Adam(predict_model.parameters(),lr=0.001)


def train(dataloader : DataLoader,model : MLP,loss_F,optim):
    model.train()
    for batch, (x,y) in enumerate(dataloader):
        x,y = x.to(device),y.to(device)
        predict = model(x)
        optim.zero_grad()
        loss = loss_F(predict,y)
        loss.backward()
        optim.step()
        if batch % 500 ==0 :
            loss = loss.item()
            print("第{}批 ： loss = {}".format(batch,loss))

def test(dataloader : DataLoader,model : MLP,loss_F):
    model.eval()
    current = 0
    batch_all = 0
    with torch.no_grad():
        for batch, (x,y) in enumerate(dataloader):
            x,y = x.to(device),y.to(device)
            predict = model(x) #predict的形状是64✖10的
            loss = loss_F(predict,y)
            # if batch % 10 == 0:
            #     print(f"test_loss{loss}")
            current += (predict.argmax(dim = 1) == y).float().sum().item()
            batch_all += 64
    print(f"总的准确度{current/batch_all}")



for i in range(10):
    print(f"{i}/10 :")
    train(train_dataload,predict_model,loss,optims)
    test(train_dataload,predict_model,loss)
print("done!")