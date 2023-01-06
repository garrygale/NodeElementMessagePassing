# -*- coding: utf-8 -*-
"""
Last version @ Nov.27, 2022,
@author: Rui "Garry" Gao

node-elem-sin for Airfoil data set.
Node stage option B

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
import math


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


    elementNodeFeatureMean = torch.mean(elementNodeFeature,dim=(0,1),keepdim=True)
    elementNodeFeatureStd = torch.std(elementNodeFeature,dim=(0,1),keepdim=True)

    elementFeatureMean = torch.mean(elementFeature,dim=0,keepdim=True)
    elementFeatureStd = torch.std(elementFeature,dim=0,keepdim=True)

    elementNodeFeatureU = (elementNodeFeature-elementNodeFeatureMean)/elementNodeFeatureStd
    elementFeatureU = (elementFeature-elementFeatureMean)/elementFeatureStd

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

class elemBlock(nn.Module):
    """
    This only applies for the simplest node-element GNN, with the same element throughout the graph
    """
    def __init__(self,nin,nElementNode,nElement,nout,hidden_dim):
        super().__init__()
        self.elementMLP = siren(nin+nElementNode+nElement,nout,hidden_dim)

    def forward(self,nodes,elementNodes,elements,elemConnGather,maxNelem):
        gatheredNodes = torch.gather(input=nodes.unsqueeze(-2).expand(-1,-1,maxNelem,-1),dim=1,
                          index=elemConnGather,sparse_grad=False)
        out = self.elementMLP(torch.cat((gatheredNodes,elementNodes,
                                         elements.unsqueeze(-2).expand(-1,-1,maxNelem,-1)),dim=-1))
        return torch.mean(out,dim=-2,keepdim=False)

class nodeBlock(nn.Module):
    """
    This only applies for the simplest node-element GNN.
    """
    def __init__(self,nin,nElement,nout,hidden_dim):
        super().__init__()
        self.nodeMLP = siren(nin+nElement,nout,hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self,nodes,elements,elemConnScatter,maxNelem):
        aggElements = torch_scatter.scatter(elements.unsqueeze(-2).expand(-1,-1,maxNelem,-1).reshape([nodes.shape[0],-1,self.hidden_dim]),
                                            elemConnScatter,dim=1,
                                            dim_size=nodes.shape[1],reduce='sum')
        return self.nodeMLP(torch.cat((aggElements,nodes),dim=-1))

class graphNetV1(nn.Module):
    """
    Node-element message passing
    """
    def __init__(self,node_dim,elementNode_dim,element_dim,out_dim,hidden_dim,nLayers):
        super().__init__()
        self.nodeEncoder = siren(node_dim,hidden_dim,hidden_dim)
        self.elemNodeEncoder = siren(elementNode_dim,hidden_dim,hidden_dim)
        self.elemEncoder = siren(element_dim,hidden_dim,hidden_dim)
        self.nLayers = nLayers
        self.elemBlks = nn.ModuleList([])
        self.nodeBlks = nn.ModuleList([])
        for i in range(nLayers):
            self.elemBlks.append(elemBlock(hidden_dim,hidden_dim,hidden_dim,hidden_dim,hidden_dim))
            self.nodeBlks.append(nodeBlock(hidden_dim,hidden_dim,hidden_dim,hidden_dim))

        self.nodeDecoder = siren(hidden_dim,out_dim,hidden_dim)

    def forward(self,nodes,elemNodes,elems,elemConn):
        nodes = self.nodeEncoder(nodes)
        elemNodes = self.elemNodeEncoder(elemNodes)
        elems = self.elemEncoder(elems)

        maxNelem = elemConn.shape[-1]
        elemConnGather = elemConn.unsqueeze(-1).expand(-1,-1,-1,nodes.shape[-1])
        elemConnScatter = elemConn.permute((0,2,1)).reshape((elemConn.shape[0],-1)).unsqueeze(-1)

        for i in range(self.nLayers):
            elems = elems+self.elemBlks[i](nodes,elemNodes,elems,elemConnGather,maxNelem) # incorporating a residual link
            nodes = nodes+self.nodeBlks[i](nodes,elems,elemConnScatter,maxNelem)
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


    trainMax = torch.amax(trainData,dim=(0,2),keepdim=True)
    trainMin = torch.amin(trainData,dim=(0,2),keepdim=True)

    trainInU = 2*(trainIn-trainMin)/(trainMax-trainMin)-1
    trainInU = torch.cat((trainInU,CBCfeatures.unsqueeze(-1).expand(-1,-1,trainInU.shape[2])),dim=1)

    trainOut = 2*trainOut/(trainMax-trainMin)


    outputMax = torch.amax(trainOut,dim=(0,2),keepdim=True)
    outputMin = torch.amin(trainOut,dim=(0,2),keepdim=True)

    dOutput = outputMax-outputMin

    outputMax = outputMax + 0.01*dOutput # so that the training output is not [-1,1] but (-1,1)
    outputMin = outputMin - 0.01*dOutput

    trainOutU = 2*(trainOut-outputMin)/(outputMax-outputMin)-1

    trainInU = trainInU.permute((2,0,1))
    trainOutU = trainOutU.permute((2,0,1))

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

        testInThisRe = 2*(upcrdCs[...,(sampleNeachRe+maxTrainRoll):(sampleNeachRe+maxTrainRoll+nStepRollout)]-trainMin
                          )/(trainMax-trainMin)-1
        testInThisRe = torch.cat((testInThisRe,CBCfeatures.unsqueeze(-1).expand(-1,-1,testInThisRe.shape[2])),dim=1)
        if j==0:
            testInU = testInThisRe.permute((2,0,1)).unsqueeze(0)
        else:
            testInU = torch.cat((testInU,testInThisRe.permute((2,0,1)).unsqueeze(0)),dim=0)

        testOutThisRe = upcrdCs[...,(sampleNeachRe+maxTrainRoll+1):(sampleNeachRe+maxTrainRoll+nStepRollout+1)]-upcrdCs[...,(sampleNeachRe+maxTrainRoll):(sampleNeachRe+maxTrainRoll+nStepRollout)]
        testOutThisRe = 2*(2*testOutThisRe/(trainMax-trainMin)-outputMin)/(outputMax-outputMin)-1

        if j==0:
            testOutU = testOutThisRe.permute((2,0,1)).unsqueeze(0)
        else:
            testOutU = torch.cat((testOutU,testOutThisRe.permute((2,0,1)).unsqueeze(0)),dim=0)

    del testInThisRe,testOutThisRe,upcrdCs
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
            loss = F.smooth_l1_loss(predNodesInc,useOut[...,:3],beta=math.sqrt(emaBeta))
            with torch.no_grad():
                varErr = (F.mse_loss(predNodesInc,useOut[...,:3])).item()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if varErr<emaBeta:
                    emaBeta = emaBeta*(1.-1./stdWindow)+1./stdWindow*varErr

            if ii%992==991:
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
            useElemNodes = elementNodeFeatureU.expand((batchSize,-1,-1,-1)).to(device)
            useElems = elementFeatureU.expand((batchSize,-1,-1)).to(device)
            useElemConn = elementCnnC.expand((batchSize,-1,-1)).to(device)

        for ii in range(nIter):
            useNodes = trainInUep[ii*batchSize:(ii+1)*batchSize,...].to(device=device)
            useOut = trainOutUep[ii*batchSize:(ii+1)*batchSize,...].to(device=device)

            optimizer.zero_grad()
            predNodesInc = network(useNodes,useElemNodes,useElems,useElemConn)
            loss = F.smooth_l1_loss(predNodesInc,useOut[...,:3],beta=math.sqrt(emaBeta))
            with torch.no_grad():
                varErr = (F.mse_loss(predNodesInc,useOut[...,:3])).item()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if varErr<emaBeta:
                    emaBeta = emaBeta*(1.-1./stdWindow)+1./stdWindow*varErr

            if ii%992==991:
                print('epoch '+str(epoch),' loss ',loss.item(),' betaSqr ',emaBeta, 'mse', varErr)

        endTime = time.time()
        print('time per Epoch: ', int(endTime-sttTime),' sec')

    return emaBeta

# %%

network = graphNetV1(node_dim=7,elementNode_dim=2,element_dim=2,out_dim=3,hidden_dim=128,nLayers=15)
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

# torch.save(network.state_dict(),'nodeElem_B_sin_Airfoil_NoNoise.pt')

network.cpu()
network.load_state_dict(torch.load('nodeElem_B_sin_Airfoil_NoNoise.pt'))
network.to(device)
network.eval()

# %%
nStepRollout = 900
nSttStep = 0
nStepInt = 10

nSaveSteps = int(nStepRollout/nStepInt)

r2Save = np.zeros([nSaveSteps,testInU.shape[0]],dtype=np.float32)
network.eval()


outputMaxU = outputMax.permute((2,0,1)).to(device)
outputMinU = outputMin.permute((2,0,1)).to(device)
trainMaxU = trainMax.permute((2,0,1))
trainMinU = trainMin.permute((2,0,1))

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
            useNodes[...,:3] += (predNodesInc+1)*(outputMaxU-outputMinU)/2+outputMinU

        timeC = time.time()

        NNtime += timeB-timeA
        otherTime += timeC-timeB

        useNodesBack = (useNodes[...,:3].cpu()).squeeze().detach()
        useNodesBack = ((useNodesBack+1)*(trainMaxU-trainMinU)/2+trainMinU)/500.

        useNodesBackGT = (testInU[j,[ii+1+nSttStep],:,:3].cpu()).squeeze().detach()
        useNodesBackGT = ((useNodesBackGT+1)*(trainMaxU-trainMinU)/2+trainMinU)/500.

        if ii%nStepInt == (nStepInt-1):
            r2Save[int((ii-nStepInt+1)/nStepInt),j] = 1-torch.mean((useNodesBack[...,2]-useNodesBackGT[...,2])**2
                                                          )/torch.mean((torch.mean(useNodesBackGT[...,2])-useNodesBackGT[...,2])**2)

        # we plot out the field at 250, 500, 750 time steps
        # if ii%250 == 249:
        #     if testRe[j] == 1900 or testRe[j] == 2450 or testRe[j] == 3100:
        #         torch.save(useNodesBack,'airfoil_nodeElem_sin_'+str(ii+1)+'_'+str(testRe[j].item())+'.pt')

print('NN time per step ',NNtime/nStepRollout/testInU.shape[0])
print('overhead per step ',otherTime/nStepRollout/testInU.shape[0])

# NN time per step  0.015354273902045357
# overhead per step  0.00014899503743207013

# %%

plotX = np.asarray(np.arange(nSttStep,nStepRollout+1,nStepInt))

for j in range(testRe.shape[0]):
    plt.figure()
    plt.plot(plotX,np.concatenate(([1],r2Save[:,j]),axis=0),label=' ')
    plt.legend(loc='lower left')
    plt.title('node-elem MP sin, airfoil, Re '+str(testRe[j].item()))
    plt.ylim([0.95,1])
    plt.xlim([0,1000])

# torch.save(r2Save,'airfoil_nodeElem_sin.pt')






