from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
from model.moe_tools import GCNConv_moe
from grb.utils.normalize import GCNAdjNorm
from torch_geometric.nn.norm import BatchNorm, InstanceNorm, LayerNorm, GraphNorm, GraphSizeNorm, PairNorm, MessageNorm, DiffGroupNorm
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=128):
        super(GCN, self).__init__()
        # self.norm_layer_1 = nn.BatchNorm1d(in_channels)
        self.adj_norm_func = GCNAdjNorm
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.norm_layer_2 = nn.BatchNorm1d(hidden_dim)
        self.conv_mid = GCNConv(hidden_dim, hidden_dim)
        self.norm_layer_3 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_channels)
        # self.norm_layer_1_output = None
        self.norm_layer_2_output = None
        self.norm_layer_3_output = None


    def forward(self, x, pos_edge_index):
        # x = self.norm_layer_1(x)
        # self.norm_layer_1_output = x
        x = self.conv1(x, pos_edge_index)
        x = self.norm_layer_2(x)
        self.norm_layer_2_output = x
        x = self.conv_mid(x, pos_edge_index)
        x = self.norm_layer_3(x)
        self.norm_layer_3_output = x
        x = self.conv2(x, pos_edge_index)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits, edge_index

class GCN_moe(GCN):
    def __init__(self, in_channels, out_channels, hidden_dim=128, num_experts=4, noisy_gating=True, k=1, dropout=0.0, use_batch_norm=False, expert_diversity= False):
        super(GCN_moe, self).__init__(in_channels=in_channels, out_channels=out_channels)
        self.conv1 = GCNConv_moe(in_channels, hidden_dim, num_experts=num_experts, noisy_gating=noisy_gating, k=k)
        self.conv_mid = GCNConv_moe(hidden_dim, hidden_dim, num_experts=num_experts, noisy_gating=noisy_gating, k=k)
        self.conv2 = GCNConv_moe(hidden_dim, out_channels, num_experts=num_experts, noisy_gating=noisy_gating, k=k)
        self.use_batch_norm = use_batch_norm
        self.expert_diversity = expert_diversity
        self.norm_layer_1_output = None
        self.norm_layer_2_output = None
        self.norm_layer_3_output = None
        if use_batch_norm:
            self.norm_layer_1 = GraphNorm(in_channels)
            self.norm_layer_2 = GraphNorm(hidden_dim)
            self.norm_layer_3 = GraphNorm(hidden_dim)
            # self.norm_layer_1 = nn.BatchNorm1d(in_channels)
            # self.norm_layer_2 = nn.BatchNorm1d(hidden_dim)
            # self.norm_layer_3 = nn.BatchNorm1d(hidden_dim)

        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
    def forward(self, x, pos_edge_index):
        if self.use_batch_norm:
            expert_list1 = []
            expert_list2 = []
            if self.expert_diversity:
                x = self.norm_layer_1(x)
                self.norm_layer_1_output = x
                x, expert_list1 = self.conv1(x, pos_edge_index)
                output1 = x
                x = self.norm_layer_2(x)
                self.norm_layer_2_output = x
                ############BN plot purpose #############
                # x, expert_list3 = self.conv_mid(x, pos_edge_index)
                # x = self.norm_layer_3(x)
                ############BN plot purpose #############
                x, expert_list2 = self.conv2(x, pos_edge_index)
                output2 = x
                if self.dropout is not None:
                    x = self.dropout(x)
                return x, output1, output2, expert_list1, expert_list2
            x = self.norm_layer_1(x)
            self.norm_layer_1_output = x
            x, expert_list1 = self.conv1(x, pos_edge_index)
            output1 = x
            x = self.norm_layer_2(x)
            self.norm_layer_2_output = x
            ############BN plot purpose #############
            # x, expert_list3 = self.conv_mid(x, pos_edge_index)
            # x = self.norm_layer_3(x)
            ############BN plot purpose #############
            x, expert_list2 = self.conv2(x, pos_edge_index)
            output2 = x
        else:
            x = self.conv1(x, pos_edge_index)
            output1 = x
            # x = self.conv_mid(x, pos_edge_index)
            x = self.conv2(x, pos_edge_index)
            output2 = x
        if self.dropout is not None:
            x = self.dropout(x)
        return x, output1, output2

