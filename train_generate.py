import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import argparse, os
import random
import re

import time
import os

# Train on CPU (hide GPU) due to memory constraints
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize
from input_data import generate_data
from model import OurModelVAE, OurModelAE, GCNModelAE, GCNModelVAE
from preprocessing import construct_feed_dict, sparse_to_tuple
from preprocessing import mask_test_graphs, graph_padding, preprocess_graph_generate_e, preprocess_graph_generate

# Settings
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
    parser.add_argument('--hidden3', type=int, default=32, help='Number of units in hidden layer 3.')
    parser.add_argument('--hidden4', type=int, default=16, help='Number of units in hidden layer 4.')
    parser.add_argument('--weight_decay', type=float, default=0., help='Weight for L2 loss on embedding matrix.')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
    parser.add_argument('--vae', type=int, default=1, help='1 for variational objective') ## here is for the graphite

    parser.add_argument('--model', type=str, default='our_vae', help='Model string.') #our_ae,our_vae,gcn_ae,gcn_vae
    parser.add_argument('--dataset', type=str, default='Erdos_Renyi', help='Dataset string.') #Erdos_Renyi,Ego,Regular,Geometric,Power_Law,Barabasi_Albert
    parser.add_argument('--features', type=int, default=1, help='Whether to use features (1) or not (0).')
    parser.add_argument('--num_graph', type=int, default=300, help='Whether to use features (1) or not (0).')
    parser.add_argument('--graph_max_size', type=int, default=20, help='Whether to use features (1) or not (0).')
    ## the flags for graphite
    parser.add_argument('--autoregressive_scalar', type=float, default=0., help='Scalar for Graphite')

    return parser.parse_args()

def get_roc_score(features, adj_norm, adj_label, adj_orig, size, model, model_str='our_vae', emb=None):
    """
    get the ROC score ,AP score, reconstruction error and neg log-likelyhood
    adj_norm:  the normalized adjs to calculated. It is a tuple for feed_dict.
    adj_label: the A+I to calculate. It is a tuple for feed_dict.
    adj_orig: the A+I as label of the adj.
    size: the size of the graphs. size <= max_size
    """
    def sigmoid(x):
        return .5 * (1 + np.tanh(.5 * x))

    with torch.no_grad():
        size = int(size)
        if emb is None:
            if model_str == 'our_vae':
                neg_log_like, _, emb = model.forward_reconstruction(features, adj_norm, adj_label)
            elif model_str == 'gcn_vae':
                neg_log_like, _, emb = model.forward_val(features, adj_norm, adj_label)
            else:
                neg_log_like, _, emb = model(features, adj_norm, adj_label)
        if model_str.startswith('our') or model_str.startswith('gcn'):
            adj_rec = np.dot(emb.to('cpu'), emb.to('cpu').T)
        preds_all = adj_rec[:size, :size]
        preds_all = preds_all.flatten()
        preds_all = sigmoid(preds_all)
        labels_all = adj_orig[:size, :size]
        labels_all = labels_all.flatten()
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)
        ## reconstruction err
        recons_error = np.mean(np.abs(labels_all - preds_all))
        ## the RMSE
        rmse = np.sqrt(np.mean(np.square(labels_all - preds_all)))
        return roc_score, ap_score, recons_error, neg_log_like.item(), rmse

args = parse_config()
model_str = args.model
dataset_str = args.dataset
num_graph = args.num_graph
graph_max_size = args.graph_max_size
epochs = args.epochs

vae = args.vae

seed = 133 # 133
np.random.seed(seed)
torch.manual_seed(seed)
graph_list, graph_size = generate_data(dataset_str, num_graph, [graph_max_size, graph_max_size], seed = seed)
graph_list, max_size = graph_padding(graph_list, graph_size)
features = sp.identity(int(max_size))

## split the train and valid and test set
graph_train, train_size,graph_val,val_size, graph_test, test_size = mask_test_graphs(graph_list, graph_size)
if model_str.startswith("our"):
    adj_norms, adj_labels = preprocess_graph_generate_e(graph_train)
    adj_norms_val, adj_labels_val = preprocess_graph_generate_e(graph_val)
    adj_norms_test, adj_labels_test = preprocess_graph_generate_e(graph_test)
else:
    adj_norms, adj_labels = preprocess_graph_generate(graph_train)
    adj_norms_val, adj_labels_val = preprocess_graph_generate(graph_val)
    adj_norms_test, adj_labels_test = preprocess_graph_generate(graph_test)

num_nodes = max_size
features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create model
adj = graph_list[0]
pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
if model_str == 'gcn_ae':
    model = GCNModelAE(args, num_features, num_nodes, features_nonzero, args.dropout, pos_weight, device)
elif model_str == 'our_vae':
    model = OurModelVAE(args, num_features, num_nodes, features_nonzero, args.dropout, pos_weight, device)
elif model_str == 'our_ae':
    model = OurModelAE(args, num_features, num_nodes, features_nonzero, args.dropout, pos_weight, device)
elif model_str == 'gcn_vae':
    model = GCNModelVAE(args, num_features, num_nodes, features_nonzero, args.dropout, pos_weight, device)

model = model.to(device)
print(model)

cost_val = []
acc_val = []
val_roc_score = []

if model_str.startswith("our_vae"):
    epochs = int(epochs/2)


# Train model
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# optimizer = torch.optim.SGD(model.parameters(), lr=1)
for epoch in range(epochs):
    ## enumerate the graphs
    for idx in range(len(adj_norms)):
        ## construct the adj_norm and adj_label
        adj_norm = adj_norms[idx]
        adj_label = adj_labels[idx]
        t = time.time()
        if model_str.startswith("our_vae"):
            # Run the construction_part
            sub_iter = 2
            sub_loss = 0
            sub_loss_num = 0
            sub_pre_loss = 1e4
            outs = 0
            for i in range(sub_iter):
                outs = model.forward_reconstruction(features, adj_norm, adj_label)
                sub_loss += outs[0]
                sub_loss_num += 1
                model.zero_grad()
                outs[0].backward()
                optimizer.step()
                if sub_iter % 15 == 0:
                    sub_loss_mean = sub_loss / sub_loss_num
                    if sub_pre_loss - sub_loss_mean< 1e-2:
                        sub_pre_loss = sub_loss_mean
                        print(sub_iter)
                        break
                    sub_pre_loss = sub_loss_mean
                    sub_loss = sub_loss_num = 0
        ##Run the ELBO
        outs = model(features, adj_norm, adj_label)
        model.zero_grad()
        outs[0].backward()
        optimizer.step()
    # Compute average loss
    avg_cost = outs[0]
    avg_accuracy = outs[1]
    ## the test process on valid set
    val_roc_score_temp = []
    ap_curr_temp = []
    recon_error_temp = []
    log_like_temp = []
    for idx, adj in enumerate(graph_val):
        adj_orig = adj + np.identity(adj.shape[0])
        roc_curr, ap_curr, recon_error, log_like, rmse = get_roc_score(features, adj_norms_val[idx], adj_labels_val[idx], adj_orig, val_size[idx], model, model_str)
        val_roc_score_temp.append(roc_curr)
        ap_curr_temp.append(ap_curr)
        recon_error_temp.append(recon_error)
        log_like_temp.append(log_like)
    val_roc_score.append(np.mean(val_roc_score_temp))
    ap_curr = np.mean(ap_curr_temp)
    recon_err = np.mean(recon_error_temp)
    log_like = np.mean(log_like_temp)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost.item()),
          "train_acc=", "{:.5f}".format(avg_accuracy.item()), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
          "val_ap=", "{:.5f}".format(ap_curr),
          "reconstruction error=", "{:.5f}".format(recon_err),
          "log_like=", "{:.5f}".format(log_like),
          "RMSE=", "{:.5f}".format(rmse),
          "time=", "{:.5f}".format(time.time() - t))

print("Optimization Finished!")
roc_test = []
ap_test = []
recon_err = []
log_like = []
for idx, adj in enumerate(graph_test):
    adj_orig = adj + np.identity(adj.shape[0])
    roc_score_temp, ap_score_temp, recon_err_temp, log_like_temp, rmse = get_roc_score(features, adj_norms_test[idx],adj_labels_test[idx],adj_orig, test_size[idx], model, model_str)
    roc_test.append(roc_score_temp)
    ap_test.append(ap_score_temp)
    recon_err.append(recon_err_temp)
    log_like.append(log_like_temp)
print('Test ROC score: ' + str(np.mean(roc_test)))
print('Test AP score: ' + str(np.mean(ap_test)))
print("Test Reconstruction error:" + str(np.mean(recon_err)))
print("Test Log Likelihood:" + str(np.mean(log_like)))
print("Test RMSE:" + str(np.mean(rmse)))
