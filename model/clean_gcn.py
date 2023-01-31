import numpy as np
import re
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from torch.nn.parameter import Parameter
# from grb.model.torch import GCN
from torch.nn.modules.module import Module
from grb.model.torch.gcn import GCNConv
from grb.utils.normalize import GCNAdjNorm


torch.manual_seed(0)

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def load_data(path="../data/", dataset="cora"):
    # Load citation network dataset
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(2000)
    idx_test = range(2000, 2708)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_test


def normalize(mx):
    # Row-normalize sparse matrix
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    # Convert a scipy sparse matrix to a torch sparse tensor
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class GraphConvolution(Module):
    #Simple GCN layer: https://arxiv.org/abs/1609.02907
    def __init__(self, in_features, out_features, bias = True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        matmul = torch.matmul(input, self.weight)
        spmm = torch.spmm(adj, matmul)
        if self.bias is not None:
            spmm+=self.bias
        return spmm


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        # self.norm_layer_1 = nn.BatchNorm1d(nfeat)
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.norm_layer_2 = nn.BatchNorm1d(nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.norm_layer_3 = nn.BatchNorm1d(nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        # self.norm_layer_3 = nn.BatchNorm1d(nhid)
        self.dropout = dropout


    def forward(self, x, adj):
        # x = F.dropout(x, p= self.dropout, training= self.training)
        # x = self.norm_layer_1(x)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, p= self.dropout, training= self.training)
        x = self.norm_layer_2(x)
        x = F.relu(self.gc3(x, adj))
        x = self.norm_layer_3(x)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


#Chunhui's version, Comment the GCN first (line 116) and then uncomment this part
# class gcn_bn(GCN):
#     def __init__(self,in_features,
#                  out_features,
#                  hidden_features,
#                  n_layers,
#                  activation=F.relu,
#                  layer_norm=False,
#                  residual=False,
#                  feat_norm=None,
#                  adj_norm_func=GCNAdjNorm,
#                  dropout=0.0):
#         super(gcn_bn, self).__init__(in_features=in_features, out_features=out_features, hidden_features=hidden_features, n_layers=n_layers, activation=activation, layer_norm=layer_norm, residual=residual, feat_norm=feat_norm, adj_norm_func=adj_norm_func, dropout=dropout)
#         self.in_features = in_features
#         self.out_features = out_features
#         self.feat_norm = feat_norm
#         self.adj_norm_func = adj_norm_func
#         if type(hidden_features) is int:
#             hidden_features = [hidden_features] * (n_layers - 1)
#         elif type(hidden_features) is list or type(hidden_features) is tuple:
#             assert len(hidden_features) == (n_layers - 1), "Incompatible sizes between hidden_features and n_layers."
#         n_features = [in_features] + hidden_features + [out_features]
#
#         self.layers = nn.ModuleList()
#         for i in range(n_layers):
#             if layer_norm:
#                 if i != 0:
#                     self.layers.append(nn.BatchNorm1d(n_features[i]))
#             self.layers.append(GCNConv(in_features=n_features[i],
#                                        out_features=n_features[i + 1],
#                                        activation=activation if i != n_layers - 1 else None,
#                                        residual=residual if i != n_layers - 1 else False,
#                                        dropout=dropout if i != n_layers - 1 else 0.0))
#         self.reset_parameters()
#
#     def forward(self, x, adj):
#         for layer in self.layers:
#             if isinstance(layer, nn.BatchNorm1d):
#                 x = layer(x)
#             else:
#                 x = layer(x, adj)
#         return x