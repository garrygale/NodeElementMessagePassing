# -*- coding: utf-8 -*-
"""
Last version @ Nov.26, 2022, 23:52 PT
@author: Rui "Garry" Gao

MeshGraphNet baseline for Airfoil data set.

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
    """
    Graph message-passing with edge carrying information, i.e., deepMind graph Net without global blocks
    """
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

    edgeFeatureMean = torch.mean(edgeFeature,dim=0,keepdim=True)
    edgeFeatureStd = torch.std(edgeFeature,dim=0,keepdim=True)
    edgeFeatureU = (edgeFeature-edgeFeatureMean)/edgeFeatureStd
    edgeFeatureU = edgeFeatureU.unsqueeze(0)
    edgesCnnC = edgesCnnC.unsqueeze(0)

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

            if ii%992==991:
                print('epoch '+str(epoch),' loss ',loss.item())

        endTime = time.time()
        print('time per Epoch: ', int(endTime-sttTime),' sec')

# %%

network = graphNet(node_dim=7,edge_dim=3,out_dim=3,hidden_dim=128,nLayers=15)
network.to(device)


# optimizer = optim.Adam(network.parameters(), lr=1e-3,betas=(0.9,0.999))

# trainStt = time.time()

# warmup(network,batchSize = 4,sttlr=1e-6,endlr=1e-4,nEpoch=10)
# train(network,batchSize = 4,learningRate=1e-4,nEpoch=40,lrDecay=1.)
# train(network,batchSize = 4,learningRate=1e-4,nEpoch=100,lrDecay=.955)
# train(network,batchSize = 4,learningRate=1e-6,nEpoch=50,lrDecay=1.)

# trainEnd = time.time()
# print('total training time', int(trainEnd-trainStt),' sec')

# torch.save(network.state_dict(),'baseline_Airfoil_NoNoise.pt')

network.cpu()
network.load_state_dict(torch.load('baseline_Airfoil_NoNoise.pt'))
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

useSenders = edgesCnnC[...,[0]].to(device)
useReceivers = edgesCnnC[...,[1]].to(device)
useEdges = edgeFeatureU.to(device=device,dtype=torch.float32)

NNtime = 0.
otherTime = 0.

for j in range(testInU.shape[0]):
    useNodes = testInU[j,[nSttStep],...].to(device=device,dtype=torch.float32)
    predNodesInc = network(useNodes,useEdges,useSenders,useReceivers) # to initialize the graph for the first time

    for ii in range(nStepRollout):
        timeA = time.time()
        predNodesInc = network(useNodes,useEdges,useSenders,useReceivers)
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
        #         torch.save(useNodesBack,'airfoil_nodeEdge_relu_'+str(ii+1)+'_'+str(testRe[j].item())+'.pt')

print('NN time per step ',NNtime/nStepRollout/testInU.shape[0])
print('overhead per step ',otherTime/nStepRollout/testInU.shape[0])

# NN time per step  0.014057013891361377
# overhead per step  9.303183908815738e-05


# %%

plotX = np.asarray(np.arange(nSttStep,nStepRollout+1,nStepInt))

for j in range(testRe.shape[0]):
    plt.figure()
    plt.plot(plotX,np.concatenate(([1],r2Save[:,j]),axis=0),label=' ')
    plt.legend(loc='lower left')
    plt.title('baseline, airfoil, Re '+str(testRe[j].item()))
    plt.ylim([0.95,1])
    plt.xlim([0,500])

# torch.save(r2Save,'airfoil_nodeEdge_relu.pt')






