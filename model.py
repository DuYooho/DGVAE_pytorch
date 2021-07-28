from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np

def weighted_cross_entropy_with_logits(logits, targets, pos_weight):
    import pdb; pdb.set_trace()
    return targets * -logits.sigmoid().log() * pos_weight + (1 - targets) * -(1 - logits.sigmoid()).log()

class GCNModelVAE(nn.Module):
    def __init__(self, args, num_features, num_nodes, features_nonzero, dropout, pos_weight, device):
        super(GCNModelVAE, self).__init__()

        self.args = args
        self.device = device
        self.input_dim = num_features # {int} 19
        self.features_nonzero = features_nonzero # {int} 19
        self.n_samples = num_nodes # {int} 19
        self.pos_weight = pos_weight # {float}
        self.GCN_sparse = GraphConvolutionSparse(self.input_dim, self.args.hidden1, dropout, self.device)
        self.GCN = GraphConvolution(self.args.hidden1, self.args.hidden2, dropout, self.device)
        self.GCN_std = GraphConvolution(self.args.hidden1, self.args.hidden2, dropout, self.device)
        self.IPDecoder = InnerProductDecoder(dropout, self.device)

    def forward(self, features, adj, labels):
        hidden1 = self.GCN_sparse(features, adj)
        z_mean = self.GCN(hidden1, adj, lambda x: x)
        z_std = self.GCN_std(hidden1, adj, lambda x: x)
        z = z_mean + torch.randn([self.n_samples, self.args.hidden2]).to(self.device) * torch.exp(z_std)
        reconstructions = self.IPDecoder(z, lambda x: x)

        labels = torch.sparse_coo_tensor(indices=torch.from_numpy(labels[0].transpose()), values=labels[1], size=labels[2]).to(torch.float32)
        labels = torch.reshape(labels.to_dense(), [-1]).to(self.device)
        # labels = labels.to_dense().to(self.device)

        cost = torch.mean(nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight))(reconstructions,labels))
        log_like = cost

        kl = (0.5 / self.n_samples) * torch.mean(torch.sum(1 + 2 * z_std - torch.square(z_mean) - torch.square(torch.exp(z_std)), 1))

        cost -= kl
        correct_prediction = torch.eq((torch.greater_equal(torch.sigmoid(reconstructions), 0.5)).to(torch.int32), labels.to(torch.int32))
        accuracy = torch.mean(correct_prediction.to(torch.float32))

        return cost, accuracy, z_mean, log_like

    def forward_val(self, features, adj, labels):
        hidden1 = self.GCN_sparse(features, adj)
        z_mean = self.GCN(hidden1, adj, lambda x: x)
        z_std = self.GCN_std(hidden1, adj, lambda x: x)
        z = z_mean + torch.randn([self.n_samples, self.args.hidden2]).to(self.device) * torch.exp(z_std)
        reconstructions = self.IPDecoder(z, lambda x: x)

        labels = torch.sparse_coo_tensor(indices=torch.from_numpy(labels[0].transpose()), values=labels[1], size=labels[2]).to(torch.float32)
        labels = torch.reshape(labels.to_dense(), [-1]).to(self.device)
        # labels = labels.to_dense().to(self.device)

        cost = torch.mean(nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight))(reconstructions,labels))

        correct_prediction = torch.eq((torch.greater_equal(torch.sigmoid(reconstructions), 0.5)).to(torch.int32), labels.to(torch.int32))
        accuracy = torch.mean(correct_prediction.to(torch.float32))

        return cost, accuracy, z_mean

class GCNModelAE(nn.Module):
    def __init__(self, args, num_features, num_nodes, features_nonzero, dropout, pos_weight, device):
        super(GCNModelAE, self).__init__()

        self.args = args
        self.device = device
        self.input_dim = num_features # {int} 19
        self.features_nonzero = features_nonzero # {int} 19
        self.n_samples = num_nodes # {int} 19
        self.pos_weight = pos_weight # {float}
        self.GCN_sparse = GraphConvolutionSparse(self.input_dim, self.args.hidden1, dropout, self.device)
        self.GCN = GraphConvolution(self.args.hidden1, self.args.hidden2, dropout, self.device)
        self.IPDecoder = InnerProductDecoder(dropout, self.device)

    def forward(self, features, adj, labels):
        hidden1 = self.GCN_sparse(features, adj)
        z_mean = self.GCN(hidden1, adj, lambda x: x)
        reconstructions = self.IPDecoder(z_mean, lambda x: x)

        labels = torch.sparse_coo_tensor(indices=torch.from_numpy(labels[0].transpose()), values=labels[1],
                                         size=labels[2]).to(torch.float32)
        labels = torch.reshape(labels.to_dense(), [-1]).to(self.device)
        cost = torch.mean(nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight))(reconstructions, labels))
        correct_prediction = torch.eq((torch.greater_equal(torch.sigmoid(reconstructions), 0.5)).to(torch.int32),
                                      labels.to(torch.int32))
        accuracy = torch.mean(correct_prediction.to(torch.float32))

        return cost, accuracy, z_mean

class OurModelAE(nn.Module):
    def __init__(self, args, num_features, num_nodes, features_nonzero, dropout, pos_weight, device):
        super(OurModelAE, self).__init__()

        self.args = args
        self.device = device
        self.input_dim = num_features # {int} 19
        self.features_nonzero = features_nonzero # {int} 19
        self.n_samples = num_nodes # {int} 19
        self.pos_weight = pos_weight # {float}
        self.GCN_sparse = GraphConvolutionSparse(self.input_dim, self.args.hidden1, dropout, self.device)
        self.GCN = GraphConvolution(self.args.hidden1, self.args.hidden2, dropout, self.device)
        self.IPDecoder = InnerProductDecoder(dropout, self.device)

    def forward(self, features, adj, labels):
        hidden1 = self.GCN_sparse(features, adj)
        z_mean = self.GCN(hidden1, adj, lambda x: x)
        reconstructions = self.IPDecoder(z_mean, lambda x: x)

        labels = torch.sparse_coo_tensor(indices=torch.from_numpy(labels[0].transpose()), values=labels[1],
                                         size=labels[2]).to(torch.float32)
        labels = torch.reshape(labels.to_dense(), [-1]).to(self.device)
        cost = torch.mean(nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight))(reconstructions, labels))
        correct_prediction = torch.eq((torch.greater_equal(torch.sigmoid(reconstructions), 0.5)).to(torch.int32),
                                      labels.to(torch.int32))
        accuracy = torch.mean(correct_prediction.to(torch.float32))

        return cost, accuracy, z_mean

class OurModelVAE(nn.Module):
    def __init__(self, args, num_features, num_nodes, features_nonzero, dropout, pos_weight, device):
        super(OurModelVAE, self).__init__()

        self.args = args
        self.device = device
        self.input_dim = num_features # {int} 19
        self.features_nonzero = features_nonzero # {int} 19
        self.n_samples = num_nodes # {int} 19
        self.pos_weight = pos_weight # {float}
        self.GCN_sparse = GraphConvolutionSparse(self.input_dim, self.args.hidden1, dropout, self.device)
        self.GCN = GraphConvolution(self.args.hidden1, self.args.hidden2, dropout, self.device)
        self.GCN_std = GraphConvolution(self.args.hidden1, self.args.hidden2, dropout, self.device)
        self.IPDecoder = InnerProductDecoder(dropout, self.device)

    def forward(self, features, adj, labels):
        hidden1 = self.GCN_sparse(features, adj)
        z_mean = self.GCN(hidden1, adj, lambda x: x)
        z_std = self.GCN_std(hidden1, adj, lambda x: x)
        z = z_mean + torch.randn([self.n_samples, self.args.hidden2]).to(self.device) * torch.sqrt(torch.exp(z_std))
        reconstructions = self.IPDecoder(z, lambda x: x)

        labels = torch.sparse_coo_tensor(indices=torch.from_numpy(labels[0].transpose()), values=labels[1], size=labels[2]).to(torch.float32)
        labels = torch.reshape(labels.to_dense(), [-1]).to(self.device)

        cost = torch.mean(nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight))(reconstructions,labels))

        a = 0.01 * np.ones((1, self.args.hidden2)).astype(np.float32)
        mu2 = torch.tensor((np.log(a).T - np.mean(np.log(a), -1)).T)
        var2 = torch.tensor((((1.0 / a) * (1 - (2.0 / self.args.hidden2))).T +
                                    (1.0 / (self.args.hidden2 * self.args.hidden2)) * np.sum(1.0 / a, -1)).T)
        mu2 = mu2.to(self.device)
        var2 = var2.to(self.device)
        ## the KL loss for the c
        latent_loss = 1 * (torch.sum(torch.div(torch.exp(z_std), var2), -1) + \
                           torch.sum(torch.multiply(torch.div((mu2 - z_mean), var2),
                                                    (mu2 - z_mean)), -1) - self.args.hidden2 + \
                           torch.sum(torch.log(var2), -1) - torch.sum(z_std, -1))
        kl = 0.5 / self.n_samples * torch.mean(latent_loss)

        cost += 1 * kl
        correct_prediction = torch.eq((torch.greater_equal(torch.sigmoid(reconstructions), 0.5)).to(torch.int32), labels.to(torch.int32))
        accuracy = torch.mean(correct_prediction.to(torch.float32))

        return cost, accuracy, z_mean

    def forward_reconstruction(self, features, adj, labels):
        hidden1 = self.GCN_sparse(features, adj)
        z_mean = self.GCN(hidden1, adj, lambda x: x)
        z_std = self.GCN_std(hidden1, adj, lambda x: x)
        z = z_mean + torch.randn([self.n_samples, self.args.hidden2]).to(self.device) * torch.sqrt(torch.exp(z_std))
        reconstructions = self.IPDecoder(z, lambda x: x)

        labels = torch.sparse_coo_tensor(indices=torch.from_numpy(labels[0].transpose()), values=labels[1], size=labels[2]).to(torch.float32)
        labels = torch.reshape(labels.to_dense(), [-1]).to(self.device)

        cost = torch.mean(nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight))(reconstructions,labels))
        # cost = F.binary_cross_entropy_with_logits(reconstructions, labels, pos_weight = self.pos_weight *labels)
        #cost = weighted_cross_entropy_with_logits(logits=reconstructions, targets=labels, pos_weight=self.pos_weight)
        correct_prediction = torch.eq((torch.greater_equal(torch.sigmoid(reconstructions), 0.5)).to(torch.int32), labels.to(torch.int32))
        accuracy = torch.mean(correct_prediction.to(torch.float32))

        return cost, accuracy, z_mean




