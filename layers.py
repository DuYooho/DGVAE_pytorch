import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np

class GraphConvolution(nn.Module):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, dropout, device):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = Parameter(torch.Tensor(input_dim, output_dim))
        self.dropout = dropout
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        init_range = np.sqrt(6.0 / (self.input_dim + self.output_dim))
        nn.init.uniform_(self.weights, -init_range, init_range)

    def forward(self, inputs, adj, act=nn.ReLU()):
        x = inputs
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.matmul(x, self.weights)
        x = x.to(self.device)
        adj = torch.sparse_coo_tensor(indices=torch.from_numpy(adj[0].transpose()), values=torch.from_numpy(adj[1]),
                                      size=adj[2]).to(torch.float32).to_dense()
        adj = adj.to(self.device)
        x = torch.matmul(adj, x)
        outputs = act(x)
        return outputs


class GraphConvolutionSparse(nn.Module):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, dropout, device):
        super(GraphConvolutionSparse, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = Parameter(torch.Tensor(input_dim, output_dim))
        self.dropout = dropout
        # self.issparse = True
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        init_range = np.sqrt(6.0 / (self.input_dim + self.output_dim))
        nn.init.uniform_(self.weights, -init_range, init_range)

    def forward(self, inputs, adj, act=nn.ReLU()):
        x = inputs
        x = torch.sparse_coo_tensor(indices=torch.from_numpy(x[0].transpose()), values=torch.from_numpy(x[1]), size=x[2]).to(torch.float32)
        x = x.to(self.device)
        x = torch.matmul(x.to_dense(), self.weights)
        x = F.dropout(x, self.dropout, training=self.training)
        adj = torch.sparse_coo_tensor(indices=torch.from_numpy(adj[0].transpose()), values=torch.from_numpy(adj[1]),
                                size=adj[2]).to(torch.float32).to_dense()
        adj = adj.to(self.device)
        x = torch.matmul(adj, x)
        outputs = act(x)
        return outputs

class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""
    def __init__(self, dropout, device):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.device = device

    def forward(self, inputs, act=nn.Sigmoid()):
        inputs = F.dropout(inputs, self.dropout, training=self.training)
        x = inputs.transpose(1,0)
        x = torch.matmul(inputs, x)
        x = torch.reshape(x, [-1])
        outputs = act(x)
        outputs = outputs.to(self.device)
        return outputs




