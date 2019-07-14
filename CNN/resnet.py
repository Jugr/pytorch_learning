import torch
from torch import nn
from torch.nn import functional as F


class ResBlk(nn.Module):
    def __init__(self,ch_in,ch_out):
        """
        :param ch_in:
        :param ch_out:
        """

        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            #[b,ch_in,h,w] => [b,ch_out,h,w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1),
                nn.BatchNorm2d(ch_out)
            )


    def forward(self,x):
        """
        :param x:
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(x))
        #shortcut  前提是通道相等
        out += self.extra(x)

        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32)
        )

        self.blk1 = ResBlk(32,64)
        self.blk2 = ResBlk(64,64)
        # self.blk3 = ResBlk(128,256)
        # self.blk4 = ResBlk(256,512)

        self.outlayer = nn.Linear(64*32*32,10)

    def forward(self , x):
        """
        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))
        #[b,64,h,w] => [b,1024,h,w]
        x = self.blk1(x)
        x = self.blk2(x)
        # x = self.blk3(x)
        # x = self.blk4(x)

        x = x.view(x.size(0),-1)
        x = self.outlayer(x)

        return x

def main():
    # tmp = torch.rand(2,64,32,32)
    # blk = ResBlk(64,128)
    # out = blk(tmp)
    # print(out.shape)

    model = ResNet18()
    tmp = torch.randn(2,3,32,32)
    out = model(tmp)
    print(out.shape)


if __name__ == '__main__':
    main()