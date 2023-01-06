# -*- coding: utf-8 -*-
"""
Last version @ Dec. 25, 2022, 16:27 PT
@author: Rui "Garry" Gao

node-elem for Airfoil data set.
Node stage option A

See the notebooks for cylinder flow data set for more info.



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

# Data from different Re share the same base grid, and therefore share
# Edge connectivity matrix; edge features; base coordinates; boundary conditions.

fileIn = h5py.File('NACA0012NNmesh.mat','r')
cnnC = fileIn['conn'] # connectivity
cnnC = np.array(cnnC,dtype=np.int64).transpose()-1
crdC = fileIn['crd'] # coordinates
crdC = np.array(crdC,dtype=np.float32).transpose()
CBCcyl = fileIn['BCCyl'] # boundary of airfoil
CBCcyl = np.unique(np.array(CBCcyl,dtype=np.int64).transpose())-1
CBCleft = fileIn['BCLeft']
CBCleft = np.unique(np.array(CBCleft,dtype=np.int64).transpose())-1
CBCright = fileIn['BCRight']
CBCright = np.unique(np.array(CBCright,dtype=np.int64).transpose())-1
CBCtop = fileIn['BCTop']
CBCtop = np.unique(np.array(CBCtop,dtype=np.int64).transpose())-1
CBCbottom = fileIn['BCBottom']
CBCbottom = np.unique(np.array(CBCbottom,dtype=np.int64).transpose())-1
fileIn.close()


cnnC = torch.from_numpy(cnnC)
crdC = torch.from_numpy(crdC)

# %% Pre-processing
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

# Same as that of the cylinder case
with torch.no_grad():
    CBCfeatures = torch.zeros([crdC.shape[0],4],dtype=torch.float32)
    CBCfeatures[CBCleft,0] = 1.  # left boundary
    CBCfeatures[CBCright,1] = 1. # right boundary
    CBCy0 = np.setdiff1d(np.unique(np.concatenate((CBCleft,CBCright,CBCtop,CBCbottom))),np.concatenate((CBCleft,CBCright)))
    CBCfeatures[CBCy0,2] = 1. # top and bottom boundary that do not overlap with left or right boundary
    CBCfeatures[CBCcyl,3] = 1. # cylinder

# Train and test Re.
trainRe = torch.cat((torch.arange(2000,3100,100),torch.arange(2033,3033,100),torch.arange(2067,3067,100)))
testRe = torch.cat((torch.arange(1000,2000,100),torch.arange(2050,3050,100),torch.arange(3100,4100,100)))
trainRe = torch.unique(trainRe)
testRe = torch.unique(testRe)


maxTrainRoll = 10 # max steps used for training the flow propagation
sampleNeachRe = 512
sampleN = sampleNeachRe*trainRe.shape[0]
nStepRollout = 1001


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

# %% Separate train & test, input and output

# Data structure. For the training set, flow data from different Re will be directly stacked in the sample dim.
# The same stacking strategy will also be applied to the rollout outputs

with torch.no_grad():
    trainOutsMultStep = []

    for j in range(trainRe.shape[0]):
        upcrdCs = torch.load('upCAirfoilRe'+str(trainRe[j].item())+'.pt')
        upcrdCs = upcrdCs[:,:3,:]

        trainDataThisRe = upcrdCs[...,:(sampleNeachRe+2)]
        if j==0:
            trainData = trainDataThisRe
        else:
            trainData = torch.cat((trainData,trainDataThisRe),dim=-1)

        trainInThisRe = trainDataThisRe[...,1:-1]
        if j==0:
            trainIn = trainInThisRe
        else:
            trainIn = torch.cat((trainIn,trainInThisRe),dim=-1)

        trainOutThisRe = trainDataThisRe[...,2:]-trainDataThisRe[...,1:-1]
        if j==0:
            trainOut = trainOutThisRe
        else:
            trainOut = torch.cat((trainOut,trainOutThisRe),dim=-1)

    trainMean = torch.mean(trainData,dim=(0,2),keepdim=True)
    trainStd = torch.std(trainData,dim=(0,2),keepdim=True)

    trainInU = (trainIn-trainMean)/trainStd
    trainInU = torch.cat((trainInU,CBCfeatures.unsqueeze(-1).expand(-1,-1,trainInU.shape[2])),dim=1)

    trainOut = trainOut/trainStd

    outputMean = torch.mean(trainOut,dim=(0,2),keepdim=True)
    outputStd = torch.std(trainOut,dim=(0,2),keepdim=True)

    trainOutU = (trainOut-outputMean)/outputStd

    trainInU = trainInU.permute((2,0,1))
    trainOutU = trainOutU.permute((2,0,1))

    elementNodeFeatureMean = torch.mean(elementNodeFeature,dim=(0,1),keepdim=True)
    elementNodeFeatureStd = torch.std(elementNodeFeature,dim=(0,1),keepdim=True)

    elementFeatureMean = torch.mean(elementFeature,dim=0,keepdim=True)
    elementFeatureStd = torch.std(elementFeature,dim=0,keepdim=True)

    elementNodeFeatureU = (elementNodeFeature-elementNodeFeatureMean)/elementNodeFeatureStd
    elementFeatureU = (elementFeature-elementFeatureMean)/elementFeatureStd

    elementNodeFeatureU = elementNodeFeatureU.unsqueeze(0)
    elementFeatureU = elementFeatureU.unsqueeze(0)
    elementCnnC = elementCnnC.unsqueeze(0)

    del trainData,trainIn,trainOut,trainDataThisRe,trainInThisRe,trainOutThisRe

    # end training set
    # test set
    # For the test data set, another dimension will be created, as we will test on different Re separately.

    for j in range(testRe.shape[0]):
        upcrdCs = torch.load('upCAirfoilRe'+str(testRe[j].item())+'.pt')
        upcrdCs = upcrdCs[:,:3,:]

        testInThisRe = (upcrdCs[...,(sampleNeachRe+maxTrainRoll):(sampleNeachRe+maxTrainRoll+nStepRollout)]-trainMean)/trainStd
        testInThisRe = torch.cat((testInThisRe,CBCfeatures.unsqueeze(-1).expand(-1,-1,testInThisRe.shape[2])),dim=1)
        if j==0:
            testInU = testInThisRe.permute((2,0,1)).unsqueeze(0)
        else:
            testInU = torch.cat((testInU,testInThisRe.permute((2,0,1)).unsqueeze(0)),dim=0)

        testOutThisRe = upcrdCs[...,(sampleNeachRe+maxTrainRoll+1):(sampleNeachRe+maxTrainRoll+nStepRollout+1)]-upcrdCs[...,(sampleNeachRe+maxTrainRoll):(sampleNeachRe+maxTrainRoll+nStepRollout)]
        testOutThisRe = ((testOutThisRe)/trainStd - outputMean)/outputStd

        if j==0:
            testOutU = testOutThisRe.permute((2,0,1)).unsqueeze(0)
        else:
            testOutU = torch.cat((testOutU,testOutThisRe.permute((2,0,1)).unsqueeze(0)),dim=0)

    del testInThisRe,testOutThisRe,upcrdCs
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

            if ii%992==991:
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

            if ii%992==991:
                print('epoch '+str(epoch),' loss ',loss.item())

        endTime = time.time()
        print('time per Epoch: ', int(endTime-sttTime),' sec')

# %%

network = graphNetV2(node_dim=7,elementNode_dim=2,element_dim=2,out_dim=3,hidden_dim=128,nLayers=15)
network.to(device)


# optimizer = optim.Adam(network.parameters(), lr=1e-3,betas=(0.9,0.999))

# trainStt = time.time()

# warmup(network,batchSize = 4,sttlr=1e-6,endlr=1e-4,nEpoch=10)
# train(network,batchSize = 4,learningRate=1e-4,nEpoch=40,lrDecay=1.)
# train(network,batchSize = 4,learningRate=1e-4,nEpoch=100,lrDecay=.955)
# train(network,batchSize = 4,learningRate=1e-6,nEpoch=50,lrDecay=1.)

# trainEnd = time.time()
# print('total training time', int(trainEnd-trainStt),' sec')

# torch.save(network.state_dict(),'nodeElem_A_Airfoil_NoNoise.pt')

network.cpu()
network.load_state_dict(torch.load('nodeElem_A_Airfoil_NoNoise.pt'))
network.to(device)
network.eval()


# %%
nStepRollout = 900
nSttStep = 0
nStepInt = 10

nSaveSteps = int(nStepRollout/nStepInt)

r2Save = np.zeros([nSaveSteps,testInU.shape[0]],dtype=np.float32)
network.eval()

outputMeanU = outputMean.permute((2,0,1)).to(device)
outputStdU = outputStd.permute((2,0,1)).to(device)
trainMeanU = trainMean.permute((2,0,1))
trainStdU = trainStd.permute((2,0,1))

useElemNodes = elementNodeFeatureU.to(device)
useElems = elementFeatureU.to(device)
useElemConn = elementCnnC.to(device)

NNtime = 0.
otherTime = 0.

for j in range(testInU.shape[0]):
    useNodes = testInU[j,[nSttStep],...].to(device=device,dtype=torch.float32)
    predNodesInc = network(useNodes,useElemNodes,useElems,useElemConn) # to initialize the graph for the first time

    for ii in range(nStepRollout):
        timeA = time.time()
        predNodesInc = network(useNodes,useElemNodes,useElems,useElemConn)
        timeB = time.time()
        with torch.no_grad():
            useNodes[...,:3] += (predNodesInc*outputStdU+outputMeanU)

        timeC = time.time()

        NNtime += timeB-timeA
        otherTime += timeC-timeB

        useNodesBack = (useNodes[...,:3].cpu()).squeeze().detach()
        useNodesBack = (useNodesBack*trainStdU+trainMeanU)/500.

        useNodesBackGT = (testInU[j,[ii+1+nSttStep],:,:3].cpu()).squeeze().detach()
        useNodesBackGT = (useNodesBackGT*trainStdU+trainMeanU)/500.

        if ii%nStepInt == (nStepInt-1):
            r2Save[int((ii-nStepInt+1)/nStepInt),j] = 1-torch.mean((useNodesBack[...,2]-useNodesBackGT[...,2])**2
                                                          )/torch.mean((torch.mean(useNodesBackGT[...,2])-useNodesBackGT[...,2])**2)

        # we plot out the field at 250, 500, 750 time steps
        # if ii%250 == 249:
        #     if testRe[j] == 1900 or testRe[j] == 2450 or testRe[j] == 3100:
        #         torch.save(useNodesBack,'airfoil_nodeElemAlt_gelu_'+str(ii+1)+'_'+str(testRe[j].item())+'.pt')

print('NN time per step ',NNtime/nStepRollout/testInU.shape[0])
print('overhead per step ',otherTime/nStepRollout/testInU.shape[0])

# NN time per step  0.014534677355377761
# overhead per step  0.00010026344546565302

# %%

plotX = np.asarray(np.arange(nSttStep,nStepRollout+1,nStepInt))

for j in range(testRe.shape[0]):
    plt.figure()
    plt.plot(plotX,np.concatenate(([1],r2Save[:,j]),axis=0),label=' ')
    plt.legend(loc='lower left')
    plt.title('node-elem MP, airfoil, Re '+str(testRe[j].item()))
    plt.ylim([0.95,1])
    plt.xlim([0,500])

# torch.save(r2Save,'airfoil_nodeElemAlt_gelu.pt')







