import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets ,transforms

batch_size = 200
learning_rate = 0.01
epochs = 10

training_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data',train=True,download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,),(0.3081,))
                   ])),
    batch_size=batch_size,shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data',train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,),(0.3081,))
                   ])),
    batch_size=batch_size,shuffle=True
)



class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784,200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200,200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200,10),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


device = torch.device('cuda:0')
net = MLP().to(device)

optimizer = optim.SGD(net.parameters(),lr=learning_rate)
criteon = nn.CrossEntropyLoss().to(device)

for epochs in range(epochs):

    for batch_idx,(data,target) in enumerate(training_loader):
        data = data.view(-1,28*28)
        data , target = data.to(device),target.cuda()

        logits = net(data)
        loss = criteon(logits,target)

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(),w2.grad.norm())
        optimizer.step()

        if batch_size %100 == 0:
            print('Train Epochs : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epochs,batch_idx * len(data) , len(training_loader.dataset),
                100.*batch_idx / len(training_loader),loss.item()
            ))


    test_loss = 0
    correct = 0
    for data,target in test_loader:
        data = data.view(-1,28*28)
        data, target = data.to(device), target.cuda()
        logits = net(data)
        test_loss += criteon(logits,target).item()

        pred = logits.data.max(1)[1]

        correct += pred.eq(target).float().sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy :{}/{} ({:.0f}%)\n'.format(
        test_loss,correct,len(test_loader.dataset),
        100.*correct /len(test_loader.dataset)
    ))
























