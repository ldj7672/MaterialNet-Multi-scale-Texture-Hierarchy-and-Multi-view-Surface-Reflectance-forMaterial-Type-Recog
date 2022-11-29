import torch
import torch.nn as nn
import torch.nn.functional as F


class depthwise_separable_conv(nn.Module): 
    def __init__(self, input_dim, output_dim): 
        super(depthwise_separable_conv, self).__init__() 
        self.depthwise = nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, groups=input_dim) 
        self.pointwise = nn.Conv2d(input_dim, output_dim, kernel_size=1) 
    def forward(self, x): 
        x = self.depthwise(x) 
        x = self.pointwise(x) 
        return x



class Normalize(nn.Module):
    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, self.p, self.dim, eps=1e-8)

def norm(input):
    return F.normalize(input,p=2.0, dim=1, eps=1e-12, out=None)

def convert2D(x,x_size):
    a = torch.zeros((len(x),2),dtype=torch.int8)
    a[:,0] = x//x_size
    a[:,1] = x%x_size
    return a

def select_salientFeature(x,numPatch,patchSize):
    patchPool = nn.AvgPool2d((patchSize,patchSize),stride=3)
    b_size,C,H,W = x.size()
    bb = patchPool(x)
    _,_,numWin,_=bb.size()
    bb = torch.max(bb,dim=1)[0]     
    bb = bb.view(b_size,-1)
    bb = torch.topk(bb,k=numPatch,dim=1)[1]
    bb = bb.view(-1)
    bb = convert2D(bb,numWin)
    patch = torch.zeros((b_size*numPatch,C,patchSize,patchSize)).to(x.device)
    for i in range(b_size*numPatch):
        a = bb[i,0]
        b = bb[i,1]
        patch[i] = x[i//numPatch,:,3*a:3*a+patchSize,3*b:3*b+patchSize]
    del x
    return patch

def MSTH_module(x,numPatch,patchSize,conv):
    avgpool = nn.AdaptiveAvgPool2d((1, 1))
    relu = nn.ReLU()
    
    B,_,_,_ = x.size()
    x_ = avgpool(relu(conv(x))).view(B,-1,1)
    x = select_salientFeature(x=x,numPatch=numPatch,patchSize=patchSize)
    x = avgpool(relu(conv(x))).view(B,numPatch,-1,1)
    if numPatch == 1 :
        x = norm(x[:,0])
    elif numPatch == 2:
        x = norm(x[:,0])*norm(x[:,1])
    elif numPatch == 3:
        x = norm(x[:,0])*norm(x[:,1])*norm(x[:,2])
    x_max = torch.max(x,dim=1)[0].view(-1,1,1)
    x_min = torch.min(x,dim=1)[0].view(-1,1,1)
    x = (x-x_min)/(x_max - x_min)
    x = x * x_
    return x

def select_salientFeature_multiview(x,numPatch,patchSize):
    patchPool = nn.AvgPool2d((patchSize,patchSize),stride=3)
    b_size,_,_,_ = x.size()
    x = patchPool(x)
    _,_,numWin,_=x.size()
    x = torch.sum(x,dim=1)
    x = x.view(b_size,-1)
    x = torch.topk(x,k=numPatch,dim=1)[1]
    x = x.view(-1)

    coordinate = convert2D(x,numWin)

    return coordinate # (b_size*numWin, 2)

def multiView_MSTH_module(x,numPatch,patchSize,conv,numView,b_size):
    avgpool = nn.AdaptiveAvgPool2d((1, 1))
    relu = nn.ReLU()
    sig = nn.Sigmoid()
    
    B,_,_,_ = x.size()
    x_entire = relu(conv(x))
    coordinate = select_salientFeature_multiview(x=x,numPatch=numPatch,patchSize=patchSize)
    salient_patch = torch.zeros((B*numPatch,x_entire.size(1),1)).to(x.device)
    for i in range(B*numPatch):
        a = coordinate[i,0]
        b = coordinate[i,1]
        salient = avgpool(x_entire[i//numPatch,:,3*a:3*a+patchSize,3*b:3*b+patchSize]).view(x_entire.size(1),1)
        salient_patch[i] = salient


    x_salient = salient_patch.view(b_size,numView,-1,1)
    x_entire = avgpool(x_entire).view(b_size,numView,-1,1)
    x_entire = torch.mean(x_entire,dim=1)

    x_max = torch.max(x_salient,dim=1)[0]
    x_mean = torch.mean(x_salient,dim=1)
    x_salient = (x_max/(x_max-x_mean+0.0001))
    x_salient = sig(x_salient)
    x_salient = x_salient * x_entire

    return x_salient
