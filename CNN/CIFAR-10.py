import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from lenet5 import LeNet5
from resnet import ResNet18
from torch import nn,optim


def main():

    batch_size = 32

    cifar_train = datasets.CIFAR10('cifar',True,transform=transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor()
        ]),download=True)

    cifar_train = DataLoader(cifar_train,batch_size=batch_size,shuffle=True)

    cifar_test = datasets.CIFAR10('cifar',False,transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]),download=True)

    cifar_test = DataLoader(cifar_test,batch_size=batch_size,shuffle=True)


    x,label = iter(cifar_train).next()
    print('x:',x.shape,'label:',label.shape)

    device = torch.device('cuda')
    # model = LeNet5().to(device)
    model = ResNet18().to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    print(model)

    for epoch in range(1000):

        model.train()
        for batchidx,(x,label) in enumerate(cifar_train):

            #[b,3,32,32]
            #[b]
            x,label = x.to(device),label.to(device)

            logits = model(x)# 针对神经网络的输入和输出
            #logits:[b,10]
            #label:[b]
            #loss:tensor scalar
            loss = criteon(logits,label)# 运用某种标准去衡量误差

            #backprop
            optimizer.zero_grad()# 将一批训练集输入之前的梯度初始化为0
            loss.backward()# 计算反向传播的梯度
            optimizer.step()#  根据梯度进行更新

        #
        print(epoch,loss.item())

        model.eval()#例如dropout什么的都恢复
        with torch.no_grad():#不需要动态图
            # test
            totol_correct = 0
            totol_num = 0
            for x,label in cifar_test:
                #[b,3,32,32]
                #[b]
                x, label = x.to(device) , label.to(device)
                #[b,10]
                logits = model(x)
                #[b]
                pred = logits.argmax(dim=1)
                totol_correct += torch.eq(pred,label).float().sum().item()
                totol_num += x.size(0)

            acc = totol_correct/totol_num
            print(epoch,acc)




if __name__ == '__main__':
    main()