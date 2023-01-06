# NodeElementMessagePassing
Code repo for the preprint "Node-element hypergraph message passing for fluid dynamics simulations".

Authors: Rui Gao, Indu Kant Deo, Rajeev K. Jaiman

https://arxiv.org/abs/2212.14545

## Requirements
PyTorch

Numpy

h5py

matplotlib (for the ploting parts within the code)

torch_scatter (if it does not work with your version of PyTorch, just copy paste its content to the code, as it should be working with a reasonably new version of PyTorch)

math

time

random

## Data sets
The data sets and trained state dicts are available at this [repo](https://drive.google.com/drive/folders/17sLVTbcDP5Y5-x4FcHumTbaBtR5xyxNj?usp=sharing).

## How to run
Each individual .py code file corresponds to one experiment reported in the paper.

To run a code file, just put the .py code file and all the data files it needs (.mat mesh file, .pt flow data, .pt saved state dict) into one folder, and set that folder as the working directory.

The files are set to be running evaluations with the provided state dict. 

To train from scratch, comment out the state dict loading part and uncomment the training part.

For Spyder users, the code is already separated into code cells. 

