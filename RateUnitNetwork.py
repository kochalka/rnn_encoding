import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class RateUnitNetwork(nn.Module):
    def __init__(self, inputSize, hiddenUnitNum, outputSize, dt, noise = None, dropUnit = None):
        super(RateUnitNetwork, self).__init__()

        self.hiddenUnitNum = hiddenUnitNum
        self.outputSize = outputSize
        self.noise = noise
        #TODO: self.tau = tau
        self.dropUnit = dropUnit
        
        self.i2h = nn.Linear(inputSize, hiddenUnitNum, bias=False)
        self.i2h.weight.data = torch.FloatTensor(torch.Size([hiddenUnitNum, inputSize])).normal_(std=1.)
        
        self.h2h = nn.Linear(hiddenUnitNum, hiddenUnitNum, bias=False)
        
        self.h2o = nn.Linear(hiddenUnitNum, outputSize, bias=False)
        self.h2o.weight.data = torch.FloatTensor(torch.Size([outputSize, hiddenUnitNum])).normal_(std = 1./np.sqrt(hiddenUnitNum))
        self.tanh = nn.Tanh()
        self.dt = dt

    def _h2h_backward_hook(self, grad):
        return grad * self.h2h_mask
            
    def init_h2h(self, prob_conn, g): #TODO: push into init()?
        size_h2h = (self.hiddenUnitNum, self.hiddenUnitNum)
        indices = np.where(np.random.uniform(0,1,size_h2h) < prob_conn)
        indices = np.delete(indices, np.where(indices[0]==indices[1]), axis=1) # remove self-connections
        mask = np.zeros(size_h2h, 'float')
        mask[indices[0],indices[1]] = 1
        self.h2h_mask = Variable(torch.from_numpy(mask).float()) # , requries_grad = False)
        indices = torch.from_numpy(indices).long()
        indices = indices.contiguous()
        sigma = g / np.sqrt(prob_conn * self.hiddenUnitNum)
        values = torch.FloatTensor(int(indices.view(-1).size(0)/2)).normal_(std=sigma)
        self.h2h.weight.data = torch.sparse.FloatTensor(indices, values, torch.Size([self.hiddenUnitNum, self.hiddenUnitNum])).to_dense()
        self.h2h.weight.register_hook(self._h2h_backward_hook)
        #return mask

    def step(self, input, hidden):
        recurrentInput = self.h2h(self.tanh(hidden))
        if self.noise == None:
            hidden = (1-self.dt)*hidden + self.dt*(self.i2h(input) + recurrentInput)
        else:
            randomVal = self.noise*np.random.uniform(-1, 1, self.hiddenUnitNum)
            randomTensor = torch.from_numpy(randomVal).float()
            randomInput = Variable(randomTensor)
            hidden = ((1-self.dt)*hidden + self.dt*(self.i2h(input)+recurrentInput))+randomInput

        if self.dropUnit != None:
            for unit in self.dropUnit:
                hidden[0, unit] = 0

        output = self.h2o(hidden)
        return output, hidden

    def forward(self, input, hidden):
        B, T, D = input.shape
        hiddens = Variable(torch.zeros(B, T, self.hiddenUnitNum))
        outputs = Variable(torch.zeros(B, T, self.outputSize))
        for t in range(T):
            outputs[:,t,:], hidden = self.step(input[:,t,:], hidden)
            hiddens[:,t,:] = hidden
        return outputs, hiddens
    
def dfs_util(a, i, v):
    v[i] = True
    c = i,
    for j in np.where(a[i])[0]:
        if not v[j]:
            c += dfs_util(a, j, v)
    return c
    
def dfs_conn_comp(a):
    N = a.shape[0]
    v = np.zeros(N, 'bool')
    c = ()
    for i, vi in enumerate(v):
        if not vi:
            c += (dfs_util(a, i, v),)
    return c

#TODO: def init_weights(func):
# self.i2h = 
# self.h2h = 
# self.h2o = 
# % if p_connect is very small, you can use WXX = sprandn(numUnits,numUnits,p_connect)*scale;
# % otherwise, use the following ("sprandn will generate significantly fewer nonzeros than requested if m*n is small or density is large")
# WXX_mask = rand(numUnits,numUnits);
# WXX_mask(WXX_mask <= p_connect) = 1;
# WXX_mask(WXX_mask < 1) = 0;
# WXX = randn(numUnits,numUnits)*scale;
# WXX = sparse(WXX.*WXX_mask);
# WXX(logical(eye(size(WXX)))) = 0;	% set self-connections to zero
# WXX_ini = WXX;

# % input connections WInputX(postsyn,presyn)
# WInputX = 1*randn(numUnits,numInputs);

# % output connections WXOut(postsyn,presyn)
# WXOut = randn(numOut,numUnits)/sqrt(numUnits);
# WXOut_ini = WXOut;

#     def forward(self, input, hidden):

#         recurrentInput = self.h2h(self.tanh(hidden))

#         if self.noise == None:
#             hidden = (1-self.dt)*hidden + self.dt*(self.i2h(input)+recurrentInput)

#         else:
#             randomVal = self.noise*np.random.uniform(-1, 1, self.hiddenUnitNum)
#             randomTensor = torch.from_numpy(randomVal).float()
#             randomInput = Variable(randomTensor)
#             hidden = ((1-self.dt)*hidden + self.dt*(self.i2h(input)+recurrentInput))+randomInput

#         if self.dropUnit != None:
#             for unit in self.dropUnit:
#                 hidden[0, unit] = 0

#         #print("Current hidden state is ", hidden)
#         output = self.h2o(hidden)
#         return output, hidden
