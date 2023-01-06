# -*- coding: utf-8 -*-
"""
Last version @ Dec. 14, 2022, 15:13 PT
@author: Rui "Garry" Gao

node-edge-sin for cylinder at Re=200

This version is with adaptive smooth L1 loss rather than the MSE loss
With sinusoidal activation function and corresponding min-max normalization, mean scatter aggregation


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
import math


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

# Pre-processing
# Split the connectivity matrix to be the edge connection matrix
with torch.no_grad():
    edgesCnnC = torch.cat((cnnC[:,[0,1]],cnnC[:,[1,2]],cnnC[:,[2,3]],cnnC[:,[3,0]]),dim=0)
    edgesCnnC = torch.unique(torch.cat((edgesCnnC,edgesCnnC[:,[1,0]]),dim=0),dim=0)

# calculate the edge feature, s_i-s_j
# option A: static input mesh encoded on edge
with torch.no_grad():
    sCrdC = torch.gather(input=crdC,dim=0,index=edgesCnnC[:,[0]].expand(-1,2))
    rCrdC = torch.gather(input=crdC,dim=0,index=edgesCnnC[:,[1]].expand(-1,2))
    edgeFeature = sCrdC-rCrdC
    edgeFeature = torch.cat((edgeFeature,torch.sqrt(edgeFeature[:,[0]]**2+edgeFeature[:,[1]]**2)),dim=1)

    edgeFeatureMean = torch.mean(edgeFeature,dim=0,keepdim=True)
    edgeFeatureStd = torch.std(edgeFeature,dim=0,keepdim=True)
    edgeFeatureU = (edgeFeature-edgeFeatureMean)/edgeFeatureStd
    edgeFeatureU = edgeFeatureU.unsqueeze(0)
    edgesCnnC = edgesCnnC.unsqueeze(0)

# Same as that of the cylinder case
with torch.no_grad():
    CBCfeatures = torch.zeros([crdC.shape[0],4],dtype=torch.float32)
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

    trainMax,_ = torch.max(trainData,dim=0,keepdim=True)
    trainMin,_ = torch.min(trainData,dim=0,keepdim=True)
    trainMax,_ = torch.max(trainMax,dim=2,keepdim=True)
    trainMin,_ = torch.min(trainMin,dim=2,keepdim=True)

    trainData = 2*(trainData-trainMin)/(trainMax-trainMin)-1

    testData = upcrdCs[...,(sampleN+maxTrainRoll):]
    testData = 2*(testData-trainMin)/(trainMax-trainMin)-1

    trainInU = trainData[...,1:(1+sampleN)]

    trainOut = trainData[...,2:(2+sampleN)]-trainInU

    outputMax,_ = torch.max(trainOut,dim=0,keepdim=True)
    outputMin,_ = torch.min(trainOut,dim=0,keepdim=True)
    outputMax,_ = torch.max(outputMax,dim=2,keepdim=True)
    outputMin,_ = torch.min(outputMin,dim=2,keepdim=True)

    dOutput = outputMax-outputMin

    outputMax = outputMax + 0.01*dOutput # so that the training output is not [-1,1] but (-1,1)
    outputMin = outputMin - 0.01*dOutput

    testInU = testData[...,:-1]
    testOut = testData[...,1:]-testInU

    trainOutU = 2*(trainOut-outputMin)/(outputMax-outputMin)-1
    testOutU = 2*(testOut-outputMin)/(outputMax-outputMin)-1

    trainInU = torch.cat((trainInU,CBCfeatures.unsqueeze(-1).expand(-1,-1,trainInU.shape[2])),dim=1)
    testInU = torch.cat((testInU,CBCfeatures.unsqueeze(-1).expand(-1,-1,testInU.shape[2])),dim=1)

    trainInU = trainInU.permute((2,0,1))
    trainOutU = trainOutU.permute((2,0,1))
    testInU = testInU.permute((2,0,1))
    testOutU = testOutU.permute((2,0,1))


# %% define network
# the code for the sine layer and the three layer mlp with sin are copied and modified from the code provided by
# Vincent Sitzmann, Julien Martel, Alexander Bergman, David Lindell, and Gordon Wetzstein. Implicit neural
# representations with periodic activation functions. Advances in Neural Information Processing Systems, 33:7462–
# 7473, 2020
class SineLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class siren(nn.Module):
    def __init__(self, nin, nout, hidden_dim,nSinLayers=2):
        super().__init__()
        self.sinA = SineLayer(nin,hidden_dim,is_first=True)
        self.sinB = nn.ModuleList()
        self.nSinLayers=nSinLayers
        if nSinLayers>1:
            for ii in range(nSinLayers-1):
                self.sinB.append(SineLayer(hidden_dim,hidden_dim))
        self.sinC = nn.Linear(hidden_dim,nout)
        with torch.no_grad():
            self.sinC.weight.uniform_(-np.sqrt(6/hidden_dim)/30,np.sqrt(6/hidden_dim)/30)

    def forward(self,x):
        out = self.sinA(x)
        if self.nSinLayers>1:
            for ii in range(self.nSinLayers-1):
                out = self.sinB[ii](out)

        return self.sinC(out)

class edgeBlock(nn.Module):
    def __init__(self,nin,nEdge,nout,hidden_dim):
        super().__init__()
        self.edgeMLP = siren(2*nin+nEdge,nout,hidden_dim)

    def forward(self,nodes,edges,senders,receivers):
        senderNodes = torch.gather(input=nodes,dim=1,index=senders.expand(-1,-1,nodes.shape[-1]),sparse_grad=False)
        receiverNodes = torch.gather(input=nodes,dim=1,index=receivers.expand(-1,-1,nodes.shape[-1]),sparse_grad=False)
        return self.edgeMLP(torch.cat((senderNodes,receiverNodes,edges),dim=-1))

class nodeBlock(nn.Module):
    def __init__(self,nin,noutPrev,nout,hidden_dim):
        super().__init__()
        self.nodeMLP = siren(nin+noutPrev,nout,hidden_dim)

    def forward(self,nodes,edges,receivers):
        aggEdges = torch_scatter.scatter(edges,receivers,dim=1,dim_size=nodes.shape[1],reduce='mean')
        return self.nodeMLP(torch.cat((nodes,aggEdges),dim=-1))

class graphNet(nn.Module):
    def __init__(self,node_dim,edge_dim,out_dim,hidden_dim,nLayers):
        super().__init__()
        self.nodeEncoder = siren(node_dim,hidden_dim,hidden_dim)
        self.edgeEncoder = siren(edge_dim,hidden_dim,hidden_dim)
        self.nLayers = nLayers
        self.edgeBlks = nn.ModuleList([])
        self.nodeBlks = nn.ModuleList([])
        for i in range(nLayers):
            self.edgeBlks.append(edgeBlock(hidden_dim,hidden_dim,hidden_dim,hidden_dim))
            self.nodeBlks.append(nodeBlock(hidden_dim,hidden_dim,hidden_dim,hidden_dim))

        self.nodeDecoder = siren(hidden_dim,out_dim,hidden_dim)

    def forward(self,nodes,edges,senders,receivers):
        nodes = self.nodeEncoder(nodes)
        edges = self.edgeEncoder(edges)
        for i in range(self.nLayers):
            edges = edges+self.edgeBlks[i](nodes,edges,senders,receivers) # incorporating a residual link
            nodes = nodes+self.nodeBlks[i](nodes,edges,receivers)
        out = self.nodeDecoder(nodes)
        return out


# %%

def warmup(network,batchSize,sttlr,endlr,nEpoch,emaBeta=1.):
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
            useEdges = edgeFeatureU.expand((batchSize,-1,-1)).to(device)
            useSenders = edgesCnnC[...,[0]].expand((batchSize,-1,-1)).to(device)
            useReceivers = edgesCnnC[...,[1]].expand((batchSize,-1,-1)).to(device)

        for ii in range(nIter):
            lrU += dlr
            for g in optimizer.param_groups:
                g['lr'] = lrU

            useNodes = trainInUep[ii*batchSize:(ii+1)*batchSize,...].to(device=device)
            useOut = trainOutUep[ii*batchSize:(ii+1)*batchSize,...].to(device=device)

            optimizer.zero_grad()
            predNodesInc = network(useNodes,useEdges,useSenders,useReceivers)
            loss = F.smooth_l1_loss(predNodesInc,useOut[...,:3],beta=math.sqrt(emaBeta))
            with torch.no_grad():
                varErr = (F.mse_loss(predNodesInc,useOut[...,:3])).item()

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if varErr<emaBeta:
                    emaBeta = emaBeta*(1.-1./stdWindow)+1./stdWindow*varErr


            if ii%128==127:
                print('epoch '+str(epoch),' loss ',loss.item(),' lr ',lrU,' betaSqr ',emaBeta, 'mse', varErr)

        endTime = time.time()
        print('time per Epoch: ', int(endTime-sttTime),' sec')

    return emaBeta

def train(network,batchSize,learningRate,nEpoch,lrDecay=1.,emaBeta=1.):
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
            useEdges = edgeFeatureU.expand((batchSize,-1,-1)).to(device)
            useSenders = edgesCnnC[...,[0]].expand((batchSize,-1,-1)).to(device)
            useReceivers = edgesCnnC[...,[1]].expand((batchSize,-1,-1)).to(device)

        for ii in range(nIter):
            useNodes = trainInUep[ii*batchSize:(ii+1)*batchSize,...].to(device=device)
            useOut = trainOutUep[ii*batchSize:(ii+1)*batchSize,...].to(device=device)
            optimizer.zero_grad()
            predNodesInc = network(useNodes,useEdges,useSenders,useReceivers)
            loss = F.smooth_l1_loss(predNodesInc,useOut[...,:3],beta=math.sqrt(emaBeta))
            with torch.no_grad():
                varErr = (F.mse_loss(predNodesInc,useOut[...,:3])).item()

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if varErr<emaBeta:
                    emaBeta = emaBeta*(1.-1./stdWindow)+1./stdWindow*varErr

            if ii%128==127:
                print('epoch '+str(epoch),' loss ',loss.item(),' betaSqr ',emaBeta, 'mse', varErr)

        endTime = time.time()
        print('time per Epoch: ', int(endTime-sttTime),' sec')

    return emaBeta

# %%

network = graphNet(node_dim=7,edge_dim=3,out_dim=3,hidden_dim=128,nLayers=15)
network.to(device)

# optimizer = optim.Adam(network.parameters(), lr=1e-3,betas=(0.9,0.999))

# stdWindow = int(sampleN/4)

# trainStt = time.time()

# emaBeta = warmup(network,batchSize = 4,sttlr=1e-6,endlr=1e-4,nEpoch=10,emaBeta=0.05)
# emaBeta = train(network,batchSize = 4,learningRate=1e-4,nEpoch=40,lrDecay=1.,emaBeta=emaBeta)
# emaBeta = train(network,batchSize = 4,learningRate=1e-4,nEpoch=100,lrDecay=.955,emaBeta=emaBeta)
# emaBeta = train(network,batchSize = 4,learningRate=1e-6,nEpoch=50,lrDecay=1.,emaBeta=emaBeta)

# trainEnd = time.time()
# print('total training time', int(trainEnd-trainStt),' sec')

# torch.save(network.state_dict(),'nodeEdge_sin_Cyl_NoNoise.pt')



# %%
network.cpu()
network.load_state_dict(torch.load('nodeEdge_sin_Cyl_NoNoise.pt'))
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
useSenders = edgesCnnC[...,[0]].to(device)
useReceivers = edgesCnnC[...,[1]].to(device)
useEdges = edgeFeatureU.to(device=device,dtype=torch.float32)

outputMaxU = outputMax.permute((2,0,1)).to(device)
outputMinU = outputMin.permute((2,0,1)).to(device)
trainMaxU = trainMax.permute((2,0,1))
trainMinU = trainMin.permute((2,0,1))
predNodesInc = network(useNodes,useEdges,useSenders,useReceivers) # to initialize the graph for the first time
NNtime = 0.
otherTime = 0.

for ii in range(nStepRollout):
    timeA = time.time()
    predNodesInc = network(useNodes,useEdges,useSenders,useReceivers)
    timeB = time.time()
    with torch.no_grad():
        useNodes[...,:3] += (predNodesInc+1)*(outputMaxU-outputMinU)/2+outputMinU

    timeC = time.time()

    NNtime += timeB-timeA
    otherTime += timeC-timeB

    # for plotting purpose
    useNodesBack = (useNodes[...,:3].cpu()).squeeze().detach()
    useNodesBack = ((useNodesBack+1)*(trainMaxU-trainMinU)/2+trainMinU)/500.

    useNodesBackGT = (testInU[[ii+1+nSttStep],:,:3].cpu()).squeeze().detach()
    useNodesBackGT = ((useNodesBackGT+1)*(trainMaxU-trainMinU)/2+trainMinU)/500.

    if ii%nStepInt == (nStepInt-1):
        r2Save[int((ii-nStepInt+1)/nStepInt),] = 1-torch.mean((useNodesBack[...,2]-useNodesBackGT[...,2])**2
                                                      )/torch.mean((torch.mean(useNodesBackGT[...,2])-useNodesBackGT[...,2])**2)

    # we plot out the field at 400, 800, 1200 time steps
    # if ii%400 == 399:
    #     torch.save(useNodesBack,'cyl_nodeEdge_sin_'+str(ii+1)+'.pt')

print('NN time per step ',NNtime/nStepRollout)
print('overhead per step ',otherTime/nStepRollout)

# NN time per step  0.01477977180480957
# overhead per step  0.00013431453704833984


# %%
plotX = np.asarray(np.arange(nStepInt,nStepRollout+1,nStepInt))

plt.figure()
plt.plot(plotX,r2Save)
plt.xlim((0,2000))
plt.ylim((0.95,1))
plt.title('node-edge MP, sin')

# %%
# torch.save(r2Save,'cyl_nodeEdge_sin.pt')
