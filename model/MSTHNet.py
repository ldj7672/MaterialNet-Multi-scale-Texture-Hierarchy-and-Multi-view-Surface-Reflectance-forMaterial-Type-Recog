import torch
import torch.nn as nn
import torchvision.models as models
from utils import *


class  MSTHNet_resnet18(nn.Module):
    def __init__(self,nclass):
        super(MSTHNet_resnet18, self).__init__()
        self.L2norm = Normalize()
        self.LSTM_input = 128       
        self.LSTM_output = 64
        self.patchSize = 10

        self.color_model = models.resnet18(pretrained=True)
        self.color_model.fc = nn.Linear(512,128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.patchPool = nn.AvgPool2d((self.patchSize,self.patchSize),stride=3)

        self.conv1 = nn.Conv2d(64,self.LSTM_input,kernel_size=3,stride=1)
        self.conv2 = nn.Conv2d(64,self.LSTM_input,kernel_size=3,stride=1)
        self.conv3 = nn.Conv2d(128,self.LSTM_input,kernel_size=3,stride=1)
        self.conv4 = nn.Conv2d(256,self.LSTM_input,kernel_size=3,stride=1)
        self.conv5 = nn.Conv2d(512,self.LSTM_input,kernel_size=3,stride=1)
        self.convlist = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
        self.convlist = nn.ModuleList(self.convlist)
        self.lstm_1 = nn.LSTM(input_size=self.LSTM_input, hidden_size=self.LSTM_output,num_layers=1,batch_first=True)
        self.fc1 = nn.Linear(128+self.LSTM_output,nclass)

        self.bn_backbone = nn.BatchNorm1d(128)
        self.bn_lstm = nn.BatchNorm1d(self.LSTM_output)

        self.relu = nn.ReLU()

    def forward(self, x):
        b_size, C, H, W = x.size()

        x = self.color_model.conv1(x)
        x = self.color_model.bn1(x)
        x = self.color_model.relu(x)
        x0 = self.color_model.maxpool(x)
        x1 = self.color_model.layer1(x0)
        x2 = self.color_model.layer2(x1)
        x3 = self.color_model.layer3(x2)
        x4 = self.color_model.layer4(x3)
        x = self.avgpool(x4).view(b_size,-1)
        x = self.color_model.fc(x)
        x = self.relu(x)
        x = self.bn_backbone(x)

        x0 = MSTH_module(x=x0, numPatch=3, patchSize=self.patchSize, conv = self.convlist[0])
        x1 = MSTH_module(x=x1, numPatch=3, patchSize=self.patchSize, conv = self.convlist[1])
        x2 = MSTH_module(x=x2, numPatch=2, patchSize=self.patchSize, conv = self.convlist[2])
        x3 = MSTH_module(x=x3, numPatch=1, patchSize=self.patchSize, conv = self.convlist[3])

        _,C,H,W = x4.size()
        x4 = self.avgpool(self.relu(self.convlist[-1](x4))).view(b_size,self.LSTM_input,1)
        x5 = torch.cat([x0,x1,x2,x3,x4],dim=2)
        del x0,x1,x2,x3,x4
        x5 = x5.permute(0,2,1)
        x5,(_,_) = self.lstm_1(x5)
        x5 = self.bn_lstm(x5[:,-1,:])

        x = self.L2norm(torch.cat([x,x5],dim = 1))
        x = self.fc1(x)

        return x

class  MSTHNet_resnet50(nn.Module):
    def __init__(self,nclass):
        super(MSTHNet_resnet50, self).__init__()
        self.L2norm = Normalize()
        self.LSTM_input =  128       
        self.LSTM_output = 64
        fc_output = 256
        self.patchSize = 10

        self.color_model = models.resnet50(pretrained=True)
        self.color_model.fc = nn.Linear(2048,fc_output)
        # for param in self.color_model.parameters(): 
        #     param.requires_grad = False
        # for param in self.color_model.fc.parameters():
        #     param.requires_grad = True
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.patchPool = nn.AvgPool2d((self.patchSize,self.patchSize),stride=3)

        self.conv1 = nn.Conv2d(64,self.LSTM_input,kernel_size= 3,stride=1)
        self.conv2 = nn.Conv2d(256,self.LSTM_input,kernel_size= 3,stride=1)
        self.conv3 = nn.Conv2d(512,self.LSTM_input,kernel_size= 3,stride=1)
        self.conv4 = nn.Conv2d(1024,self.LSTM_input,kernel_size= 3,stride=1)
        self.conv5 = nn.Conv2d(2048,self.LSTM_input,kernel_size= 3,stride=1)

        self.convlist = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
        self.convlist = nn.ModuleList(self.convlist)

        self.lstm = nn.LSTM(input_size=self.LSTM_input, hidden_size=self.LSTM_output,num_layers=1,batch_first=True)
        self.fc_final = nn.Linear(fc_output+self.LSTM_output,nclass)

        self.bn_backbone = nn.BatchNorm1d(fc_output)
        self.bn_lstm = nn.BatchNorm1d(self.LSTM_output)

        self.relu = nn.ReLU()

    def forward(self, x):
        b_size, C, H, W = x.size()

        x = self.color_model.conv1(x)
        x = self.color_model.bn1(x)
        x = self.color_model.relu(x)
        x0 = self.color_model.maxpool(x)
        x1 = self.color_model.layer1(x0)
        x2 = self.color_model.layer2(x1)
        x3 = self.color_model.layer3(x2)
        x4 = self.color_model.layer4(x3)
        x = self.avgpool(x4).view(b_size,-1)
        x = self.relu(self.bn_backbone(self.color_model.fc(x)))

        x0 = MSTH_module(x=x0, numPatch=3, patchSize=self.patchSize, conv = self.convlist[0])
        x1 = MSTH_module(x=x1, numPatch=3, patchSize=self.patchSize, conv = self.convlist[1])
        x2 = MSTH_module(x=x2, numPatch=2, patchSize=self.patchSize, conv = self.convlist[2])
        x3 = MSTH_module(x=x3, numPatch=1, patchSize=self.patchSize, conv = self.convlist[3])

        _,C,H,W = x4.size()
        x4 = self.avgpool(self.relu(self.conv5(x4))).view(b_size,self.LSTM_input,1)
        x5 = torch.cat([x0,x1,x2,x3,x4],dim=2)
        del x0,x1,x2,x3,x4
        x5 = x5.permute(0,2,1)
        x5,(_,_) = self.lstm(x5)
        x5 = self.relu(self.bn_lstm(x5[:,-1,:]))

        x = self.L2norm(torch.cat([x,x5],dim = 1))

        x = self.fc_final(x)

        return x




if __name__ == "__main__":

    model = MSTHNet_resnet50(23)
    model.eval()
    x = torch.rand((1,3,224,224))
    y = model(x)
    print(y.shape)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {pytorch_total_params}")


