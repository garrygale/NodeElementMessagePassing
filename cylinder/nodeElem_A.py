# -*- coding: utf-8 -*-
"""
Last version @ Dec. 20, 2022, 18:42 PT
@author: Rui "Garry" Gao

Node-elem hypergraph message-passing on the flow around cylinder at Reynolds number 200.
Node stage option A.
Implemented in PyTorch

GeLU activation version. We use GELU for node-elem as it performs significantly better than ReLU.
For the baseline meshGraphNet, ReLU performs significantly better than GELU, so we use ReLU.

The result produced by running the code could be different from the results in the paper, due to the use of atomic operation...
    in the scatter function. The difference, however, should be reasonably small.


"""
# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_scatter


import random
import time
import matplotlib.pyplot as plt
import numpy as np
import h5py


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


torch.backends.cudnn.deterministic = True

random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)



# %% Load data
upcrdCs = torch.load('upCRe200Long.pt')
upcrdCs = upcrdCs[:,:3,:] # u_x,u_y,p

fileIn = h5py.File('cylNNmesh.mat','r')
cnnC = fileIn['connC'] # connectivity
cnnC = np.array(cnnC,dtype=np.int64).transpose()-2 # chop off the center point of circle
crdC = fileIn['crdC'] # coordinates
crdC = np.array(crdC,dtype=np.float32).transpose()
crdC = crdC[1:,:]
CBCcyl = fileIn['CBCcyl']
CBCcyl = np.unique(np.array(CBCcyl,dtype=np.int64).transpose())-2
CBCcyl = CBCcyl[1:]
CBCleft = fileIn['Cleft']
CBCleft = np.unique(np.array(CBCleft,dtype=np.int64).transpose())-2
CBCright = fileIn['Cright']
CBCright = np.unique(np.array(CBCright,dtype=np.int64).transpose())-2
CBCtop = fileIn['Ctop']
CBCtop = np.unique(np.array(CBCtop,dtype=np.int64).transpose())-2
CBCbottom = fileIn['Cbottom']
CBCbottom = np.unique(np.array(CBCbottom,dtype=np.int64).transpose())-2
fileIn.close()

# %% Pre-processing
cnnC = torch.from_numpy(cnnC)
crdC = torch.from_numpy(crdC)

# calculate the element feature, now we just use the mean as the center of the element
# Static mesh encoded on element
with torch.no_grad():

    elementCnnC = cnnC.clone().detach()

    elementXcrds = torch.gather(input=crdC[:,[0]].expand((-1,4)),dim=0,index=elementCnnC)
    elementYcrds = torch.gather(input=crdC[:,[1]].expand((-1,4)),dim=0,index=elementCnnC)

    elementXcrdMean = torch.mean(elementXcrds,dim=1,keepdim=True)
    elementYcrdMean = torch.mean(elementYcrds,dim=1,keepdim=True)

    elementNodeFeatureX = (elementXcrds-elementXcrdMean).unsqueeze(-1)
    elementNodeFeatureY = (elementYcrds-elementYcrdMean).unsqueeze(-1)

    elementNodeFeature = torch.cat((elementNodeFeatureX,elementNodeFeatureY),dim=-1)

    elementArea = (elementXcrds[:,0]*elementYcrds[:,1]-elementXcrds[:,1]*elementYcrds[:,0]+
                   elementXcrds[:,1]*elementYcrds[:,2]-elementXcrds[:,2]*elementYcrds[:,1]+
                   elementXcrds[:,2]*elementYcrds[:,3]-elementXcrds[:,3]*elementYcrds[:,2]+
                   elementXcrds[:,3]*elementYcrds[:,0]-elementXcrds[:,0]*elementYcrds[:,3]).unsqueeze(-1)

    elementArea = torch.abs(elementArea)

    elementFeature = elementArea

    # augment it
    elementFeature = torch.cat((elementArea,-elementArea),dim=-1)

# We have actually four types of boundaries, left, top/bottom, right, and cylinder
# MeshGraphNet use a one-hot vector to denote these
# A separate dimension for normal node is not necessary, so we don't follow that part
with torch.no_grad():
    CBCfeatures = torch.zeros([upcrdCs.shape[0],4],dtype=torch.float32)
    CBCfeatures[CBCleft,0] = 1.  # left boundary
    CBCfeatures[CBCright,1] = 1. # right boundary
    CBCy0 = np.setdiff1d(np.unique(np.concatenate((CBCleft,CBCright,CBCtop,CBCbottom))),np.concatenate((CBCleft,CBCright)))
    CBCfeatures[CBCy0,2] = 1. # top and bottom boundary that do not overlap with left or right boundary
    CBCfeatures[CBCcyl,3] = 1. # cylinder


# %% Separate train & test, input and output

# To put a nice separation between train and test data set
maxTrainRoll = 10

sampleN = 2048 # total no. samples used for training
with torch.no_grad():

    trainData = upcrdCs[...,:(sampleN+2)]

    trainMean = torch.mean(trainData,dim=(0,2),keepdim=True)
    trainStd = torch.std(trainData,dim=(0,2),keepdim=True)
    trainData = (trainData-trainMean)/trainStd

    testData = upcrdCs[...,(sampleN+maxTrainRoll):]
    testData = (testData-trainMean)/trainStd # as we normalize using training data statistics

    trainInU = trainData[...,1:(1+sampleN)]

    trainOut = trainData[...,2:(2+sampleN)]-trainInU

    outputMean = torch.mean(trainOut,dim=(0,2),keepdim=True)
    outputStd = torch.std(trainOut,dim=(0,2),keepdim=True)

    trainOutU = (trainOut-outputMean)/outputStd

    testInU = testData[...,:-1]
    testOut = testData[...,1:]-testInU

    testOutU = (testOut-outputMean)/outputStd

    elementNodeFeatureMean = torch.mean(elementNodeFeature,dim=(0,1),keepdim=True)
    elementNodeFeatureStd = torch.std(elementNodeFeature,dim=(0,1),keepdim=True)

    elementFeatureMean = torch.mean(elementFeature,dim=0,keepdim=True)
    elementFeatureStd = torch.std(elementFeature,dim=0,keepdim=True)

    elementNodeFeatureU = (elementNodeFeature-elementNodeFeatureMean)/elementNodeFeatureStd
    elementFeatureU = (elementFeature-elementFeatureMean)/elementFeatureStd

    trainInU = torch.cat((trainInU,CBCfeatures.unsqueeze(-1).expand(-1,-1,trainInU.shape[2])),dim=1)
    testInU = torch.cat((testInU,CBCfeatures.unsqueeze(-1).expand(-1,-1,testInU.shape[2])),dim=1)

    trainInU = trainInU.permute((2,0,1))
    trainOutU = trainOutU.permute((2,0,1))
    testInU = testInU.permute((2,0,1))
    testOutU = testOutU.permute((2,0,1))
    elementNodeFeatureU = elementNodeFeatureU.unsqueeze(0)
    elementFeatureU = elementFeatureU.unsqueeze(0)
    elementCnnC = elementCnnC.unsqueeze(0)


# %% define network
class MLPnorm(nn.Module):
    def __init__(self, nin, nout, hidden_dim):
        super().__init__()
        self.netE = nn.Sequential(
            nn.Linear(nin, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,nout)
        )
        self.normE = nn.LayerNorm(nout)
    def forward(self, x):
        return self.normE(self.netE(x))

class MLP(nn.Module):
    def __init__(self, nin, nout, hidden_dim):
        super().__init__()
        self.netE = nn.Sequential(
            nn.Linear(nin, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,nout)
        )
    def forward(self, x):
        return self.netE(x)

class elemBlock(nn.Module):
    """
    This only applies for the simplest node-element GNN, with the same element throughout the graph
    """
    def __init__(self,nin,nElementNode,nElement,nout,hidden_dim):
        super().__init__()
        self.elementMLP = MLPnorm(nin+nElementNode+nElement,nout,hidden_dim)

    def forward(self,gatheredNodes,elementNodes,elements,elemConnGather,maxNelem):
        out = self.elementMLP(torch.cat((gatheredNodes,elementNodes,
                                         elements.unsqueeze(-2).expand(-1,-1,maxNelem,-1)),dim=-1))
        return torch.mean(out,dim=-2,keepdim=False)

class nodeBlock(nn.Module):
    """
    This only applies for the simplest node-element GNN.
    """
    def __init__(self,nin,nElementNode,nElement,nout,hidden_dim):
        super().__init__()
        self.nodeMLP = MLPnorm(nin+nElementNode+nElement,nout,hidden_dim)
        self.hidden_dim=hidden_dim

    def forward(self,gatheredNodes,elementNodes,elements,elemConnScatter,maxNelem,nodeDim,dimSize):
        interim = self.nodeMLP(torch.cat((gatheredNodes,elementNodes,
                                         elements.unsqueeze(-2).expand(-1,-1,maxNelem,-1)),dim=-1))
        aggNodeFeatures = torch_scatter.scatter(interim.reshape([nodeDim,-1,self.hidden_dim]),
                                            elemConnScatter,dim=1,
                                            dim_size=dimSize,reduce='sum')
        return aggNodeFeatures

class graphNetV2(nn.Module):
    """
    Node-element message passing, alternative
    """
    def __init__(self,node_dim,elementNode_dim,element_dim,out_dim,hidden_dim,nLayers):
        super().__init__()
        self.nodeEncoder = MLPnorm(node_dim,hidden_dim,hidden_dim)
        self.elemNodeEncoder = MLPnorm(elementNode_dim,hidden_dim,hidden_dim)
        self.elemEncoder = MLPnorm(element_dim,hidden_dim,hidden_dim)
        self.nLayers = nLayers
        self.elemBlks = nn.ModuleList([])
        self.nodeBlks = nn.ModuleList([])
        for i in range(nLayers):
            self.elemBlks.append(elemBlock(hidden_dim,hidden_dim,hidden_dim,hidden_dim,hidden_dim))
            self.nodeBlks.append(nodeBlock(hidden_dim,hidden_dim,hidden_dim,hidden_dim,hidden_dim))

        self.nodeDecoder = MLP(hidden_dim,out_dim,hidden_dim)

    def forward(self,nodes,elemNodes,elems,elemConn):
        nodes = self.nodeEncoder(nodes)
        elemNodes = self.elemNodeEncoder(elemNodes)
        elems = self.elemEncoder(elems)

        maxNelem = elemConn.shape[-1]
        nodeDim = nodes.shape[0]
        dimSize = nodes.shape[1]
        elemConnGather = elemConn.unsqueeze(-1).expand(-1,-1,-1,nodes.shape[-1])
        elemConnScatter = elemConn.permute((0,2,1)).reshape((elemConn.shape[0],-1)).unsqueeze(-1)

        for i in range(self.nLayers):
            gatheredNodes = torch.gather(input=nodes.unsqueeze(-2).expand(-1,-1,maxNelem,-1),dim=1,
                              index=elemConnGather,sparse_grad=False)
            elems = elems+self.elemBlks[i](gatheredNodes,elemNodes,elems,elemConnGather,maxNelem) # incorporating a residual link
            nodes = nodes+self.nodeBlks[i](gatheredNodes,elemNodes,elems,elemConnScatter,maxNelem,nodeDim,dimSize)
            gatheredNodes.detach_()
        out = self.nodeDecoder(nodes)
        return out

# %%
network = graphNetV2(node_dim=7,elementNode_dim=2,element_dim=2,out_dim=3,hidden_dim=128,nLayers=15)
network.to(device)



# %%

def warmup(network,batchSize,sttlr,endlr,nEpoch):
    network.train()
    nIter = int(sampleN/batchSize)
    nIterAll = nIter*nEpoch-1
    dlr = (endlr-sttlr)/nIterAll
    lrU = sttlr
    for g in optimizer.param_groups:
        g['lr'] = lrU

    for epoch in range(nEpoch):
        print('epoch ',epoch)
        sttTime = time.time()
        arrPerm = torch.randperm(sampleN)

        with torch.no_grad():
            trainInUep = trainInU[arrPerm,...].to(dtype=torch.float32)
            trainOutUep = trainOutU[arrPerm,...].to(dtype=torch.float32)
            useElemNodes = elementNodeFeatureU.expand((batchSize,-1,-1,-1)).to(device)
            useElems = elementFeatureU.expand((batchSize,-1,-1)).to(device)
            useElemConn = elementCnnC.expand((batchSize,-1,-1)).to(device)

        for ii in range(nIter):
            lrU += dlr
            for g in optimizer.param_groups:
                g['lr'] = lrU

            useNodes = trainInUep[ii*batchSize:(ii+1)*batchSize,...].to(device=device)
            useOut = trainOutUep[ii*batchSize:(ii+1)*batchSize,...].to(device=device)

            optimizer.zero_grad()
            predNodesInc = network(useNodes,useElemNodes,useElems,useElemConn)
            loss = F.mse_loss(predNodesInc,useOut[...,:3])
            loss.backward()
            optimizer.step()

            if ii%128==127:
                print('epoch '+str(epoch),' loss ',loss.item(),' lr ',lrU)

        endTime = time.time()
        print('time per Epoch: ', int(endTime-sttTime),' sec')

def train(network,batchSize,learningRate,nEpoch,lrDecay=1.):
    network.train()
    nIter = int(sampleN/batchSize)

    for epoch in range(nEpoch):
        print('epoch ',epoch)
        sttTime = time.time()

        for g in optimizer.param_groups:
            g['lr'] = learningRate*lrDecay**epoch

        arrPerm = torch.randperm(sampleN)

        with torch.no_grad():
            trainInUep = trainInU[arrPerm,...].to(dtype=torch.float32)
            trainOutUep = trainOutU[arrPerm,...].to(dtype=torch.float32)
            useElemNodes = elementNodeFeatureU.expand((batchSize,-1,-1,-1)).to(device)
            useElems = elementFeatureU.expand((batchSize,-1,-1)).to(device)
            useElemConn = elementCnnC.expand((batchSize,-1,-1)).to(device)

        for ii in range(nIter):
            useNodes = trainInUep[ii*batchSize:(ii+1)*batchSize,...].to(device=device)
            useOut = trainOutUep[ii*batchSize:(ii+1)*batchSize,...].to(device=device)

            optimizer.zero_grad()
            predNodesInc = network(useNodes,useElemNodes,useElems,useElemConn)
            loss = F.mse_loss(predNodesInc,useOut[...,:3])
            loss.backward()
            optimizer.step()

            if ii%128==127:
                print('epoch '+str(epoch),' loss ',loss.item())

        endTime = time.time()
        print('time per Epoch: ', int(endTime-sttTime),' sec')

# %%
# optimizer = optim.Adam(network.parameters(), lr=1e-3,betas=(0.9,0.999))
# trainStt = time.time()

# warmup(network,batchSize = 4,sttlr=1e-6,endlr=1e-4,nEpoch=10)
# train(network,batchSize = 4,learningRate=1e-4,nEpoch=40,lrDecay=1.)
# train(network,batchSize = 4,learningRate=1e-4,nEpoch=100,lrDecay=.955)
# train(network,batchSize = 4,learningRate=1e-6,nEpoch=50,lrDecay=1.)


# trainEnd = time.time()
# print('total training time', int(trainEnd-trainStt),' sec')

# torch.save(network.state_dict(),'nodeElemMPalt_Cyl_NoNoise.pt')


# %%
network.cpu()
network.load_state_dict(torch.load('nodeElem_A_Cyl_NoNoise.pt'))
network.to(device)
network.eval()


# %% plot out the R^2 value between the ground truth p^* and the predicted values
nStepRollout = 2000
nSttStep = 0
nStepInt = 10

nSaveSteps = int(nStepRollout/nStepInt)
r2Save = np.zeros([nSaveSteps,],dtype=np.float32)
network.eval()

useNodes = testInU[[nSttStep],...].to(device=device,dtype=torch.float32)
useElemNodes = elementNodeFeatureU.to(device)
useElems = elementFeatureU.to(device)
useElemConn = elementCnnC.to(device)

outputMeanU = outputMean.permute((2,0,1)).to(device)
outputStdU = outputStd.permute((2,0,1)).to(device)
trainMeanU = trainMean.permute((2,0,1))
trainStdU = trainStd.permute((2,0,1))
predNodesInc = network(useNodes,useElemNodes,useElems,useElemConn) # to initialize the graph for the first time
NNtime = 0.
otherTime = 0.

for ii in range(nStepRollout):
    timeA = time.time()
    predNodesInc = network(useNodes,useElemNodes,useElems,useElemConn)
    timeB = time.time()
    with torch.no_grad():
        useNodes[...,:3] += (predNodesInc*outputStdU+outputMeanU)

    timeC = time.time()

    NNtime += timeB-timeA
    otherTime += timeC-timeB

    # for plotting purpose
    useNodesBack = (useNodes[...,:3].cpu()).squeeze().detach()
    useNodesBack = (useNodesBack*trainStdU+trainMeanU)/500.

    useNodesBackGT = (testInU[[ii+1+nSttStep],:,:3].cpu()).squeeze().detach()
    useNodesBackGT = (useNodesBackGT*trainStdU+trainMeanU)/500.

    if ii%nStepInt == (nStepInt-1):
        r2Save[int((ii-nStepInt+1)/nStepInt),] = 1-torch.mean((useNodesBack[...,2]-useNodesBackGT[...,2])**2
                                                      )/torch.mean((torch.mean(useNodesBackGT[...,2])-useNodesBackGT[...,2])**2)

    # we plot out the field at 400, 800, 1200 time steps
    # if ii%400 == 399:
    #     torch.save(useNodesBack,'cyl_nodeElem_gelu_'+str(ii+1)+'.pt')

print('NN time per step ',NNtime/nStepRollout)
print('overhead per step ',otherTime/nStepRollout)

# NN time per step  0.013958209872245789
# overhead per step  0.00010156190395355224


# %%
plotX = np.asarray(np.arange(nStepInt,nStepRollout+1,nStepInt))

plt.figure()
plt.plot(plotX,r2Save)
plt.xlim((0,2000))
plt.ylim((0.95,1))
plt.title('node-elem MP A')

# %%
# torch.save(r2Save,'cyl_nodeElemAlt_gelu.pt')