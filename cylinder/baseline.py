# -*- coding: utf-8 -*-
"""
Last version @ Nov.24, 2022, 14:16 PT
@author: Rui "Garry" Gao

MeshGraphNet baseline on the flow around cylinder at Reynolds number 200.
Implemented in PyTorch following the TensorFlow code provided by Pfaff et al.

Some differences between the implementation here and the original MeshGraphNet implementation. See paper for details

This is the ReLU activation version. The baseline meshGraphNet performs better with ReLU compared with GELU, so we stick to ReLU.

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

# Split the connectivity matrix to be the edge connection matrix
with torch.no_grad():
    edgesCnnC = torch.cat((cnnC[:,[0,1]],cnnC[:,[1,2]],cnnC[:,[2,3]],cnnC[:,[3,0]]),dim=0)
    edgesCnnC = torch.unique(torch.cat((edgesCnnC,edgesCnnC[:,[1,0]]),dim=0),dim=0) # 8728.

# calculate the edge feature, s_i-s_j
# option A: static input mesh encoded on edge
with torch.no_grad():
    sCrdC = torch.gather(input=crdC,dim=0,index=edgesCnnC[:,[0]].expand(-1,2))
    rCrdC = torch.gather(input=crdC,dim=0,index=edgesCnnC[:,[1]].expand(-1,2))
    edgeFeature = sCrdC-rCrdC
    edgeFeature = torch.cat((edgeFeature,torch.sqrt(edgeFeature[:,[0]]**2+edgeFeature[:,[1]]**2)),dim=1)
    # can be added as we augment the element in node-element GNN, will not affect much
    # edgeFeature = torch.cat((edgeFeature,-edgeFeature[:,[-1]]),dim=1)

# We have actually four types of boundaries, left, top/bottom, right, and cylinder
# MeshGraphNet use a one-hot vector to denote these
# Obviously you don't need a separate dimension for normal node, so we don't follow that part
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

    edgeFeatureMean = torch.mean(edgeFeature,dim=0,keepdim=True)
    edgeFeatureStd = torch.std(edgeFeature,dim=0,keepdim=True)

    edgeFeatureU = (edgeFeature-edgeFeatureMean)/edgeFeatureStd

    trainInU = torch.cat((trainInU,CBCfeatures.unsqueeze(-1).expand(-1,-1,trainInU.shape[2])),dim=1)
    testInU = torch.cat((testInU,CBCfeatures.unsqueeze(-1).expand(-1,-1,testInU.shape[2])),dim=1)

    trainInU = trainInU.permute((2,0,1))
    trainOutU = trainOutU.permute((2,0,1))
    testInU = testInU.permute((2,0,1))
    testOutU = testOutU.permute((2,0,1))
    edgeFeatureU = edgeFeatureU.unsqueeze(0)
    edgesCnnC = edgesCnnC.unsqueeze(0)


# %% define network
class MLPnorm(nn.Module):
    def __init__(self, nin, nout, hidden_dim):
        super().__init__()
        self.netE = nn.Sequential(
            nn.Linear(nin, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,nout)
        )
    def forward(self, x):
        return self.netE(x)

class edgeBlock(nn.Module):
    def __init__(self,nin,nEdge,nout,hidden_dim):
        super().__init__()
        self.edgeMLP = MLPnorm(2*nin+nEdge,nout,hidden_dim)

    def forward(self,nodes,edges,senders,receivers):
        senderNodes = torch.gather(input=nodes,dim=1,index=senders.expand(-1,-1,nodes.shape[-1]),sparse_grad=False)
        receiverNodes = torch.gather(input=nodes,dim=1,index=receivers.expand(-1,-1,nodes.shape[-1]),sparse_grad=False)
        return self.edgeMLP(torch.cat((senderNodes,receiverNodes,edges),dim=-1))

class nodeBlock(nn.Module):
    def __init__(self,nin,noutPrev,nout,hidden_dim):
        super().__init__()
        self.nodeMLP = MLPnorm(nin+noutPrev,nout,hidden_dim)

    def forward(self,nodes,edges,receivers):
        aggEdges = torch_scatter.scatter(edges,receivers,dim=1,dim_size=nodes.shape[1],reduce='sum')
        return self.nodeMLP(torch.cat((nodes,aggEdges),dim=-1))

class graphNet(nn.Module):
    def __init__(self,node_dim,edge_dim,out_dim,hidden_dim,nLayers):
        super().__init__()
        self.nodeEncoder = MLPnorm(node_dim,hidden_dim,hidden_dim)
        self.edgeEncoder = MLPnorm(edge_dim,hidden_dim,hidden_dim)
        self.nLayers = nLayers
        self.edgeBlks = nn.ModuleList([])
        self.nodeBlks = nn.ModuleList([])
        for i in range(nLayers):
            self.edgeBlks.append(edgeBlock(hidden_dim,hidden_dim,hidden_dim,hidden_dim))
            self.nodeBlks.append(nodeBlock(hidden_dim,hidden_dim,hidden_dim,hidden_dim))

        self.nodeDecoder = MLP(hidden_dim,out_dim,hidden_dim)

    def forward(self,nodes,edges,senders,receivers):
        nodes = self.nodeEncoder(nodes)
        edges = self.edgeEncoder(edges)
        for i in range(self.nLayers):
            edges = edges+self.edgeBlks[i](nodes,edges,senders,receivers) # incorporating a residual link
            nodes = nodes+self.nodeBlks[i](nodes,edges,receivers)
        out = self.nodeDecoder(nodes)
        return out

# %%
network = graphNet(node_dim=7,edge_dim=3,out_dim=3,hidden_dim=128,nLayers=15)
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
            useEdges = edgeFeatureU.expand((batchSize,-1,-1)).to(device)
            useSenders = edgesCnnC[...,[0]].expand((batchSize,-1,-1)).to(device)
            useReceivers = edgesCnnC[...,[1]].expand((batchSize,-1,-1)).to(device)

        for ii in range(nIter):
            useNodes = trainInUep[ii*batchSize:(ii+1)*batchSize,...].to(device=device)
            useOut = trainOutUep[ii*batchSize:(ii+1)*batchSize,...].to(device=device)

            optimizer.zero_grad()
            predNodesInc = network(useNodes,useEdges,useSenders,useReceivers)
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

# torch.save(network.state_dict(),'baseline_Cyl_NoNoise.pt')

# %%
network.cpu()
network.load_state_dict(torch.load('baseline_Cyl_NoNoise.pt'))
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


outputMeanU = outputMean.permute((2,0,1)).to(device)
outputStdU = outputStd.permute((2,0,1)).to(device)
trainMeanU = trainMean.permute((2,0,1))
trainStdU = trainStd.permute((2,0,1))
predNodesInc = network(useNodes,useEdges,useSenders,useReceivers) # to initialize the graph for the first time
NNtime = 0.
otherTime = 0.

for ii in range(nStepRollout):
    timeA = time.time()
    predNodesInc = network(useNodes,useEdges,useSenders,useReceivers)
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
    #     torch.save(useNodesBack,'cyl_nodeEdge_relu_'+str(ii+1)+'.pt')

print('NN time per step ',NNtime/nStepRollout)
print('overhead per step ',otherTime/nStepRollout)
# NN time per step  0.013253718256950379
# overhead per step  6.24992847442627e-05

# %%
plotX = np.asarray(np.arange(nStepInt,nStepRollout+1,nStepInt))

plt.figure()
plt.plot(plotX,r2Save)
plt.xlim((0,2000))
plt.ylim((0.95,1))
plt.title('baseline')

# torch.save(r2Save,'cyl_nodeEdge_relu.pt')