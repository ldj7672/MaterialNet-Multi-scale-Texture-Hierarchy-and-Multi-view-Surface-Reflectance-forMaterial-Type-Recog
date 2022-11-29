import torch
import torch.nn as nn
import torchvision.models as models
from utils import *


class  MaterialNet(nn.Module):
    def __init__(self,nclass):
        super(MaterialNet, self).__init__()
        self.L2norm = Normalize()
        self.LSTM_input = 128       
        self.LSTM_output = 64
        self.patchSize = 10

        self.backboneNet = models.resnet18(pretrained=True)
        self.backboneNet.fc = nn.Linear(512,128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.patchPool = nn.AvgPool2d((self.patchSize,self.patchSize),stride=3)

        self.conv1 = nn.Conv2d(64,self.LSTM_input,kernel_size=3,stride=1)
        self.conv2 = nn.Conv2d(64,self.LSTM_input,kernel_size=3,stride=1)
        self.conv3 = nn.Conv2d(128,self.LSTM_input,kernel_size=3,stride=1)
        self.conv4 = nn.Conv2d(256,self.LSTM_input,kernel_size=3,stride=1)
        self.conv5 = nn.Conv2d(512,self.LSTM_input,kernel_size=3,stride=1)
        self.convlist = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
        self.convlist = nn.ModuleList(self.convlist)
        self.lstm_MSTH = nn.LSTM(input_size=128, hidden_size=64,num_layers=1,batch_first=True)
        self.lstm_MVSR = nn.LSTM(input_size=128, hidden_size=75,num_layers=1,batch_first=True)
        
        self.fc_final = nn.Linear(75+64,nclass)
        self.bn_MSTH = nn.BatchNorm1d(64)
        self.bn_MVSR = nn.BatchNorm1d(75)

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        if x.ndim == 4:
            b_size, C, H, W = x.size()
            numView = 1
        elif x.ndim == 5:
            b_size, numView, C, H, W = x.size()
        x = x.reshape(b_size*numView, C, H, W)
        x = self.backboneNet.conv1(x)
        x = self.backboneNet.bn1(x)
        x = self.backboneNet.relu(x)

        x0 = self.backboneNet.maxpool(x)
        x1 = self.backboneNet.layer1(x0)
        x2 = self.backboneNet.layer2(x1)
        x3 = self.backboneNet.layer3(x2)
        x4 = self.backboneNet.layer4(x3)

        x = self.avgpool(x4).view(b_size*numView,-1)

        x = self.relu(self.backboneNet.fc(x))
        x = x.view(b_size, numView, -1) 
        x,(_,_) = self.lstm_MVSR(x)
        x = self.bn_MVSR(x[:,-1,:])

        x0 = multiView_MSTH_module(x=x0,b_size = b_size,numView=numView, numPatch=1, patchSize=self.patchSize, conv = self.convlist[0])  
        x1 = multiView_MSTH_module(x=x1,b_size = b_size,numView=numView, numPatch=1, patchSize=self.patchSize, conv = self.convlist[1])   
        x2 = multiView_MSTH_module(x=x2,b_size = b_size,numView=numView, numPatch=1, patchSize=self.patchSize, conv = self.convlist[2])   
        x3 = multiView_MSTH_module(x=x3,b_size = b_size,numView=numView, numPatch=1, patchSize=self.patchSize, conv = self.convlist[3])  

        _,C,H,W = x4.size()
        x4 = self.avgpool(self.relu(self.convlist[-1](x4))).view(b_size,numView,-1,1)
        x4 = torch.mean(x4,dim=1) # view 통합
        x_MSTH = torch.cat([x0,x1,x2,x3,x4],dim=2)
        del x0,x1,x2,x3,x4
        x_MSTH = x_MSTH.permute(0,2,1)
        x_MSTH,(_,_) = self.lstm_MSTH(x_MSTH)
        x_MSTH = self.bn_MSTH(x_MSTH[:,-1,:])

        x = self.L2norm(torch.cat([x,x_MSTH],dim = 1))
        x = self.fc_final(x)

        return x

class  DualMaterialNet(nn.Module):   
    def __init__(self,nclass):
        super(DualMaterialNet, self).__init__()
        self.L2norm = Normalize()
        self.LSTM_input = 128       
        self.LSTM_output = 64
        self.patchSize = 10

        self.color_model = models.resnet18(pretrained=True)
        self.color_model.fc = nn.Linear(512,128)
        self.diff_model = models.resnet18(pretrained=True)
        self.diff_model.fc = nn.Linear(512,128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.patchPool = nn.AvgPool2d((self.patchSize,self.patchSize),stride=3)

        self.conv1 = nn.Conv2d(64,self.LSTM_input,kernel_size=3,stride=1)
        self.conv2 = nn.Conv2d(64,self.LSTM_input,kernel_size=3,stride=1)
        self.conv3 = nn.Conv2d(128,self.LSTM_input,kernel_size=3,stride=1)
        self.conv4 = nn.Conv2d(256,self.LSTM_input,kernel_size=3,stride=1)
        self.conv5 = nn.Conv2d(512,self.LSTM_input,kernel_size=3,stride=1)
        self.convlist = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
        self.convlist = nn.ModuleList(self.convlist)

        self.conv11 = nn.Conv2d(64,64,kernel_size=3,stride=1)
        self.conv22 = nn.Conv2d(64,64,kernel_size=3,stride=1)
        self.conv33 = nn.Conv2d(128,64,kernel_size=3,stride=1)
        self.conv44 = nn.Conv2d(256,64,kernel_size=3,stride=1)
        self.conv55 = nn.Conv2d(512,64,kernel_size=3,stride=1)

        self.lstm_1 = nn.LSTM(input_size=self.LSTM_input, hidden_size=self.LSTM_output,num_layers=1,batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=128, hidden_size=64,num_layers=1,batch_first=True)
        self.lstm_3 = nn.LSTM(input_size =128, hidden_size=64,num_layers=1,batch_first=True)
        self.lstm_4 = nn.LSTM(input_size =64, hidden_size=32,num_layers=1,batch_first=True)

        self.fc1 = nn.Linear(224,nclass)
        self.bn1 = nn.BatchNorm1d(self.LSTM_output)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(32)

        self.relu = nn.ReLU(inplace=True)
   
    def forward(self, x, x_diff):
        if x.ndim == 4:
            b_size, C, H, W = x.size()
            numView = 1
        elif x.ndim == 5:
            b_size, numView, C, H, W = x.size()
        x = x.reshape(b_size*numView, C, H, W)
        x = self.color_model.conv1(x)
        x = self.color_model.bn1(x)
        x = self.color_model.relu(x)

        x0 = self.color_model.maxpool(x)
        x1 = self.color_model.layer1(x0)
        x2 = self.color_model.layer2(x1)
        x3 = self.color_model.layer3(x2)
        x4 = self.color_model.layer4(x3)
        x = self.avgpool(x4).view(b_size*numView,-1)

        x = self.relu(self.color_model.fc(x))
        x = x.view(b_size, numView, -1) 
        x,(_,_) = self.lstm_2(x)
        x = self.bn2(x[:,-1,:])

        x_diff = x_diff.reshape(b_size*numView, C, H, W)
        x_diff = self.diff_model.conv1(x_diff)
        x_diff = self.diff_model.bn1(x_diff)
        x_diff = self.diff_model.relu(x_diff)

        x_diff0 = self.diff_model.maxpool(x_diff)
        x_diff1 = self.diff_model.layer1(x_diff0)
        x_diff2 = self.diff_model.layer2(x_diff1)
        x_diff3 = self.diff_model.layer3(x_diff2)
        x_diff4 = self.diff_model.layer4(x_diff3)
        x_diff = self.avgpool(x_diff4).view(b_size*numView,-1)

        x_diff = self.relu(self.diff_model.fc(x_diff))
        x_diff = x_diff.view(b_size, numView, -1) 
        x_diff,(_,_) = self.lstm_3(x_diff)
        x_diff = self.bn3(x_diff[:,-1,:])
        x0 = multiView_MSTH_module(x=x0,b_size = b_size,numView=numView, numPatch=1, patchSize=self.patchSize, conv = self.convlist[0])
        x1 = multiView_MSTH_module(x=x1,b_size = b_size,numView=numView, numPatch=1, patchSize=self.patchSize, conv = self.convlist[1])   
        x2 = multiView_MSTH_module(x=x2,b_size = b_size,numView=numView, numPatch=1, patchSize=self.patchSize, conv = self.convlist[2])   
        x3 = multiView_MSTH_module(x=x3,b_size = b_size,numView=numView, numPatch=1, patchSize=self.patchSize, conv = self.convlist[3])  

        _,C,H,W = x4.size()
        x4 = self.avgpool(self.relu(self.conv5(x4))).view(b_size,numView,-1,1)
        x4 = torch.mean(x4,dim=1)

        x_diff00 = self.avgpool(self.relu(self.conv11(x_diff0))).view(b_size,numView,-1,1)
        x_diff00 = torch.mean(x_diff00,dim=1)

        x_diff11 = self.avgpool(self.relu(self.conv22(x_diff1))).view(b_size,numView,-1,1)
        x_diff11 = torch.mean(x_diff11,dim=1)

        x_diff22 = self.avgpool(self.relu(self.conv33(x_diff2))).view(b_size,numView,-1,1)
        x_diff22 = torch.mean(x_diff22,dim=1)

        x_diff33 = self.avgpool(self.relu(self.conv44(x_diff3))).view(b_size,numView,-1,1)
        x_diff33 = torch.mean(x_diff33,dim=1)

        x_diff4 = self.avgpool(self.relu(self.conv55(x_diff4))).view(b_size,numView,-1,1)
        x_diff4 = torch.mean(x_diff4,dim=1)

        x5 = torch.cat([x0,x1,x2,x3,x4],dim=2)
        x_diff5 = torch.cat([x_diff00,x_diff11,x_diff22,x_diff33,x_diff4],dim=2)

        del x0,x1,x2,x3,x4
        x5 = x5.permute(0,2,1)
        x_diff5 = x_diff5.permute(0,2,1)

        x5,(_,_) = self.lstm_1(x5)
        x5 = self.bn1(x5[:,-1,:])

        x_diff5,(_,_) = self.lstm_4(x_diff5)
        x_diff5 = self.bn4(x_diff5[:,-1,:])

        x = self.L2norm(torch.cat([x,x_diff,x5,x_diff5],dim = 1))
        x = self.fc1(x)

        return x


if __name__ == "__main__":
    print('='*5 +'MaterialNet' + '='*5)
    model = MaterialNet(31)
    x = torch.rand((4,6,3,224,224))
    y = model(x)
    print(y.shape)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {pytorch_total_params}")

    print('='*5 +'DualMaterialNet' + '='*5)
    model = DualMaterialNet(31)
    x1 = torch.rand((4,6,3,224,224))
    x2 = torch.rand((4,6,3,224,224))
    y = model(x1, x2)
    print(y.shape)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {pytorch_total_params}")

