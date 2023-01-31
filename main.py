#!/usr/bin/env python
# coding: utf-8
import math
import os
import os.path as osp
from torch.cuda.amp import autocast
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling
from torch.nn.utils import clip_grad_norm, clip_grad_value_
from torch_geometric.nn import GCNConv
from model.diver_tools import Ncontrast, get_feature_dis
import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from model.gnn import GCN, GCN_moe
from utils import get_link_labels, prediction_fairness, Loss_contrastive, Loss_cosine, loss_mhs_weight_reg, loss_mgd_weight_reg
from torch_geometric.utils import train_test_split_edges
import argparse
import grafog.transforms as grafog_T
from utils import sparse_mx_to_torch_sparse_tensor, get_A_r
from torch_geometric.utils.convert import to_scipy_sparse_matrix

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='citeseer',)#"cora" "pubmed" "citeseer"
    argparser.add_argument('--delta', type=float, default=0.47)
    argparser.add_argument('--test_seeds', default=[42, 1337, 1234, 12345])
    argparser.add_argument('--epochs', type=int, default=500)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--out_dim', type=int, default=256, help='output dimension of GCN')
    argparser.add_argument('--hidden_dim', type=int, default=256, help='Number of hidden units of GNN.')
    argparser.add_argument('--num_experts', type=int, default=2, help='number of experts')
    argparser.add_argument('--noisy_gating', type=bool, default=True, help='whether to use noisy gating')
    argparser.add_argument('--k', type=int, default=1, help='number of experts to sample from')
    argparser.add_argument('--device', type=str, default='cuda')
    argparser.add_argument('--use_moe', type=bool, default=True, help='whether to use GNN_moe')
    argparser.add_argument('--use_aug', type=bool, default=False, help='whether to use augmentation')
    argparser.add_argument('--node_feat_mask', type=float, default=0.15, help='fraction of node features to mask')
    argparser.add_argument('--node_mixup_lamb', type=float, default=0.5, help='lambda for node mixup')
    argparser.add_argument('--edge_feat_mask', type=float, default=0.15, help='fraction of edge features to mask')
    argparser.add_argument('--edge_drop_rate', type=float, default=0.15, help='rate for edge drop')
    argparser.add_argument('--dropout', type=float, default=0.5, help='dropout rate of GNN_moe')
    argparser.add_argument('--use_bn', type=bool, default=True, help='whether to use batch normalization')
    argparser.add_argument('--use_ncontrast', type=bool, default=False, help='whether to use ncontrast')
    argparser.add_argument('--tau_ncontrast', type=float, default=1.0, help='temperature for ncontrast')
    argparser.add_argument('--delta_ncontrast', type=int, default=2, help='to compute order-th power of adj')
    argparser.add_argument('--alpha_ncontrast', type=float, default=2.0, help='To control the ratio of Ncontrast loss')
    argparser.add_argument('--cosine_loss', type= bool, default= False, help='whether to add consine loss to the last layer')
    argparser.add_argument('--contrastive_loss', type= bool, default= False, help='whether to add contrastive loss to different layers')
    argparser.add_argument('--expert_mhs_loss', type = bool, default = False, help ='whether to add mhs loss to enrich expert diversity')
    argparser.add_argument('--use_clip_norm_value', type=float, default=2.0, help='Clips gradient of an iterable of parameters at specified value')
    args = argparser.parse_args()
    acc_auc = []
    fairness = []

    # 1. Load data
    path = osp.join(osp.dirname(osp.realpath('__file__')), "..", "data", args.dataset)
    dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
    # get torch sparse adjacency matrix from torch_geometric.data.Data
    adj = dataset[0].edge_index
    scipy_adj = to_scipy_sparse_matrix(adj)
    torch_adj = sparse_mx_to_torch_sparse_tensor(scipy_adj)
    adj = torch_adj.to(args.device)
    adj_label = get_A_r(adj, args.delta_ncontrast)

    node_aug = grafog_T.Compose([
        # T.NodeDrop(p=0.45),
        grafog_T.NodeFeatureMasking(p=args.node_feat_mask),
        # grafog_T.NodeMixUp(lamb=args.node_mixup_lamb, classes=7),
    ])
    edge_aug = grafog_T.Compose([
        grafog_T.EdgeDrop(p=args.edge_drop_rate),
        # grafog_T.EdgeFeatureMasking(p=args.edge_feat_mask),
    ])

    # 2.training settings
    for random_seed in args.test_seeds:
        print("random_seed: ", random_seed)
        np.random.seed(random_seed)
        dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
        data = dataset[0]
        if args.use_aug:
            data = node_aug(data)
            data = edge_aug(data)
        protected_attribute = data.y
        # data.train_mask = data.val_mask = data.test_mask = data.y = None
        data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.2)
        data = data.to(args.device)
        num_classes = len(np.unique(protected_attribute))
        # print("num_classes: ", num_classes)
        N = data.num_nodes

        # 3. model
        if args.use_moe:
            model = GCN_moe(data.num_features, args.out_dim, hidden_dim=args.hidden_dim,
                            num_experts=args.num_experts, noisy_gating=args.noisy_gating, k=args.k,
                            dropout=args.dropout, use_batch_norm=args.use_bn, expert_diversity=args.expert_mhs_loss).to(args.device)
        else:
            model = GCN(data.num_features, args.out_dim, hidden_dim=args.hidden_dim).to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        Y = torch.LongTensor(protected_attribute).to(args.device)
        Y_aux = (Y[data.train_pos_edge_index[0, :]] != Y[data.train_pos_edge_index[1, :]]).to(args.device)
        randomization = (torch.FloatTensor(args.epochs, Y_aux.size(0)).uniform_() < 0.5 + args.delta).to(args.device)

        best_val_perf = test_perf = 0
        for epoch in range(1, args.epochs):
            # TRAINING
            neg_edges_tr = negative_sampling(
                edge_index=data.train_pos_edge_index,
                num_nodes=N,
                num_neg_samples=data.train_pos_edge_index.size(1) // 2,
            ).to(args.device)

            if epoch == 1 or epoch % 10 == 0:
                keep = torch.where(randomization[epoch], Y_aux, ~Y_aux)

            model.train()
            optimizer.zero_grad()

            if args.use_moe:
                if args.expert_mhs_loss:
                    z, output1, output2, expert_list1, expert_list2 = model(data.x, data.train_pos_edge_index[:, keep])
                else:
                    z, output1, output2 = model(data.x, data.train_pos_edge_index[:, keep])
            else:
                z = model(data.x, data.train_pos_edge_index)
            link_logits, _ = model.decode(
                z, data.train_pos_edge_index[:, keep], neg_edges_tr
            )
            tr_labels = get_link_labels(
                data.train_pos_edge_index[:, keep], neg_edges_tr
            ).to(args.device)

            contrastive_loss = 0
            consine_loss = 0
            mhs_loss = 0
            if args.contrastive_loss:
                contrastive_loss = Loss_contrastive(output1, output2)
            if args.cosine_loss:
                consine_loss = Loss_cosine(output2)
            if args.expert_mhs_loss:
                for i in expert_list1:
                    mhs_loss += loss_mhs_weight_reg(i)
                for j in expert_list2:
                    mhs_loss += loss_mhs_weight_reg(j)

            loss = F.binary_cross_entropy_with_logits(link_logits, tr_labels) + consine_loss + contrastive_loss + mhs_loss
            if args.use_ncontrast:
                x_dis = get_feature_dis(z)
                loss_cl = Ncontrast(x_dis, adj_label, tau=args.tau_ncontrast)
                loss = loss + loss_cl * args.alpha_ncontrast

            loss.backward()
            cur_loss = loss.item()
            clip_grad_value_(model.parameters(), clip_value = args.use_clip_norm_value)
            optimizer.step()

            # EVALUATION
            model.eval()
            perfs = []
            for prefix in ["val", "test"]:
                pos_edge_index = data[f"{prefix}_pos_edge_index"]
                neg_edge_index = data[f"{prefix}_neg_edge_index"]
                with torch.no_grad():
                    if args.use_moe:
                        if args.expert_mhs_loss:
                            z, output1, output2, expert_list1, expert_list2 = model(data.x, data.train_pos_edge_index)
                        else:
                            z, output1, output2 = model(data.x, data.train_pos_edge_index)
                    else:
                        z = model(data.x, data.train_pos_edge_index)

                    link_logits, edge_idx = model.decode(z, pos_edge_index, neg_edge_index)
                link_probs = link_logits.sigmoid()
                link_labels = get_link_labels(pos_edge_index, neg_edge_index)
                auc = roc_auc_score(link_labels.cpu(), link_probs.cpu())
                perfs.append(auc)

            val_perf, tmp_test_perf = perfs
            if val_perf > best_val_perf:
                best_val_perf = val_perf
                test_perf = tmp_test_perf
            if epoch % 10 == 0:
                log = "Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}"
                print(log.format(epoch, cur_loss, best_val_perf, test_perf))

        # FAIRNESS
        auc = test_perf
        cut = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        best_acc = 0
        best_cut = 0.5
        for i in cut:
            acc = accuracy_score(link_labels.cpu(), link_probs.cpu() >= i)
            if acc > best_acc:
                best_acc = acc
                best_cut = i
        f = prediction_fairness(edge_idx.cpu(), link_labels.cpu(), link_probs.cpu() >= best_cut, Y.cpu())
        acc_auc.append([best_acc * 100, auc * 100])
        fairness.append([x * 100 for x in f])

    # 3.evaluation results
    ma = np.mean(np.asarray(acc_auc), axis=0)
    mf = np.mean(np.asarray(fairness), axis=0)
    sa = np.std(np.asarray(acc_auc), axis=0)
    sf = np.std(np.asarray(fairness), axis=0)
    # 3.1performance results
    print(f"ACC: {ma[0]:2f} +- {sa[0]:2f}")
    print(f"AUC: {ma[1]:2f} +- {sa[1]:2f}")
    # 3.2fairness results
    print(f"DP mix: {mf[0]:2f} +- {sf[0]:2f}")
    print(f"EoP mix: {mf[1]:2f} +- {sf[1]:2f}")
    print(f"DP group: {mf[2]:2f} +- {sf[2]:2f}")
    print(f"EoP group: {mf[3]:2f} +- {sf[3]:2f}")
    print(f"DP sub: {mf[4]:2f} +- {sf[4]:2f}")
    print(f"EoP sub: {mf[5]:2f} +- {sf[5]:2f}")
    print("args: ", args)


#Good Version
#!/usr/bin/env python
# coding: utf-8
# import math
# import os
# import os.path as osp
# from torch_geometric.datasets import Planetoid
# import torch_geometric.transforms as T
# from torch_geometric.utils import negative_sampling
# from torch.nn.utils import clip_grad_norm, clip_grad_value_
# from torch_geometric.nn import GCNConv
# from model.diver_tools import Ncontrast, get_feature_dis
# import numpy as np
# import torch
# from torch.nn import Sequential, Linear, ReLU
# import torch.nn.functional as F
# from sklearn.metrics import roc_auc_score, accuracy_score
# from model.gnn import GCN, GCN_moe
# from utils import get_link_labels, prediction_fairness, Loss_contrastive, Loss_cosine, loss_mhs_weight_reg
# from torch_geometric.utils import train_test_split_edges
# import argparse
# import grafog.transforms as grafog_T
# from utils import sparse_mx_to_torch_sparse_tensor, get_A_r
# from torch_geometric.utils.convert import to_scipy_sparse_matrix
# if __name__ == '__main__':
#     argparser = argparse.ArgumentParser()
#     argparser.add_argument('--dataset', type=str, default='citeseer',)#"cora" "pubmed" "citeseer"
#     argparser.add_argument('--delta', type=float, default=0.20)
#     argparser.add_argument('--test_seeds', default=[1337,1234])
#     argparser.add_argument('--epochs', type=int, default=1001)
#     argparser.add_argument('--lr', type=float, default=0.001)
#     argparser.add_argument('--out_dim', type=int, default=256, help='output dimension of GCN')
#     argparser.add_argument('--hidden_dim', type=int, default=256, help='Number of hidden units of GNN.')
#     argparser.add_argument('--num_experts', type=int, default=5, help='number of experts')
#     argparser.add_argument('--noisy_gating', type=bool, default=True, help='whether to use noisy gating')
#     argparser.add_argument('--k', type=int, default=1, help='number of experts to sample from')
#     argparser.add_argument('--device', type=str, default='cuda')
#     argparser.add_argument('--use_moe', type=bool, default= True, help='whether to use GNN_moe')
#     argparser.add_argument('--use_aug', type=bool, default=False, help='whether to use augmentation')
#     argparser.add_argument('--node_feat_mask', type=float, default=0.15, help='fraction of node features to mask')
#     argparser.add_argument('--node_mixup_lamb', type=float, default=0.5, help='lambda for node mixup')
#     argparser.add_argument('--edge_feat_mask', type=float, default=0.15, help='fraction of edge features to mask')
#     argparser.add_argument('--edge_drop_rate', type=float, default=0.15, help='rate for edge drop')
#     argparser.add_argument('--dropout', type=float, default=0.5, help='dropout rate of GNN_moe')
#     argparser.add_argument('--use_bn', type=bool, default=True, help='whether to use batch normalization')
#     argparser.add_argument('--use_ncontrast', type=bool, default=True, help='whether to use ncontrast')
#     argparser.add_argument('--tau_ncontrast', type=float, default=1.0, help='temperature for ncontrast')
#     argparser.add_argument('--delta_ncontrast', type=int, default=2, help='to compute order-th power of adj')
#     argparser.add_argument('--alpha_ncontrast', type=float, default=2.0, help='To control the ratio of Ncontrast loss')
#     argparser.add_argument('--cosine_loss', type= bool, default= True, help='whether to add consine loss to the last layer')
#     argparser.add_argument('--contrastive_loss', type= bool, default= True, help='whether to add contrastive loss to different layers')
#     argparser.add_argument('--expert_mhs_loss', type = bool, default = True, help ='whether to add mhs loss to enrich expert diversity')
#     argparser.add_argument('--use_clip_norm_value', type=float, default=2.0, help='Clips gradient of an iterable of parameters at specified value')
#     args = argparser.parse_args()
#     acc_auc = []
#     fairness = []
#
#     # 1. Load data
#     path = osp.join(osp.dirname(osp.realpath('__file__')), "..", "data", args.dataset)
#     dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
#     # get torch sparse adjacency matrix from torch_geometric.data.Data
#     adj = dataset[0].edge_index
#     scipy_adj = to_scipy_sparse_matrix(adj)
#     torch_adj = sparse_mx_to_torch_sparse_tensor(scipy_adj)
#     adj = torch_adj.to(args.device)
#     adj_label = get_A_r(adj, args.delta_ncontrast)
#
#     node_aug = grafog_T.Compose([
#         # T.NodeDrop(p=0.45),
#         grafog_T.NodeFeatureMasking(p=args.node_feat_mask),
#         # grafog_T.NodeMixUp(lamb=args.node_mixup_lamb, classes=7),
#     ])
#     edge_aug = grafog_T.Compose([
#         grafog_T.EdgeDrop(p=args.edge_drop_rate),
#         # grafog_T.EdgeFeatureMasking(p=args.edge_feat_mask),
#     ])
#
#     # 2.training settings
#     for random_seed in args.test_seeds:
#         print("random_seed: ", random_seed)
#         np.random.seed(random_seed)
#         dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
#         data = dataset[0]
#         if args.use_aug:
#             data = node_aug(data)
#             data = edge_aug(data)
#         protected_attribute = data.y
#         # data.train_mask = data.val_mask = data.test_mask = data.y = None
#         data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.2)
#         data = data.to(args.device)
#         num_classes = len(np.unique(protected_attribute))
#         # print("num_classes: ", num_classes)
#         N = data.num_nodes
#
#         # 3. model
#         if args.use_moe:
#             model = GCN_moe(data.num_features, args.out_dim, hidden_dim=args.hidden_dim,
#                             num_experts=args.num_experts, noisy_gating=args.noisy_gating, k=args.k,
#                             dropout=args.dropout, use_batch_norm=args.use_bn, expert_diversity=args.expert_mhs_loss).to(args.device)
#         else:
#             model = GCN(data.num_features, args.out_dim, hidden_dim=args.hidden_dim).to(args.device)
#
#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#
#         Y = torch.LongTensor(protected_attribute).to(args.device)
#         Y_aux = (Y[data.train_pos_edge_index[0, :]] != Y[data.train_pos_edge_index[1, :]]).to(args.device)
#         randomization = (torch.FloatTensor(args.epochs, Y_aux.size(0)).uniform_() < 0.5 + args.delta).to(args.device)
#
#         best_val_perf = test_perf = 0
#         for epoch in range(1, args.epochs):
#             # TRAINING
#             neg_edges_tr = negative_sampling(
#                 edge_index=data.train_pos_edge_index,
#                 num_nodes=N,
#                 num_neg_samples=data.train_pos_edge_index.size(1) // 2,
#             ).to(args.device)
#
#             if epoch == 1 or epoch % 10 == 0:
#                 keep = torch.where(randomization[epoch], Y_aux, ~Y_aux)
#
#             model.train()
#             optimizer.zero_grad()
#
#             if args.use_moe:
#                 z, output1, output2, expert_list1, expert_list2 = model(data.x, data.train_pos_edge_index[:, keep])
#             if args.expert_mhs_loss:
#                 z, output1, output2, expert_list1, expert_list2 = model(data.x, data.train_pos_edge_index[:, keep])
#             else:
#                 z = model(data.x, data.train_pos_edge_index)
#
#             link_logits, _ = model.decode(
#                 z, data.train_pos_edge_index[:, keep], neg_edges_tr
#             )
#             tr_labels = get_link_labels(
#                 data.train_pos_edge_index[:, keep], neg_edges_tr
#             ).to(args.device)
#
#             contrastive_loss = 0
#             consine_loss = 0
#             mhs_loss = 0
#             if args.contrastive_loss:
#                 contrastive_loss = Loss_contrastive(output1, output2)
#             if args.cosine_loss:
#                 consine_loss = Loss_cosine(output2)
#             if args.expert_mhs_loss:
#                 for i in expert_list1:
#                     mhs_loss += loss_mhs_weight_reg(i)
#                 for j in expert_list2:
#                     mhs_loss += loss_mhs_weight_reg(j)
#
#             loss = F.binary_cross_entropy_with_logits(link_logits, tr_labels) + consine_loss + contrastive_loss + mhs_loss
#             if args.use_ncontrast:
#                 x_dis = get_feature_dis(z)
#                 loss_cl = Ncontrast(x_dis, adj_label, tau=args.tau_ncontrast)
#                 loss = loss + loss_cl * args.alpha_ncontrast
#
#             loss.backward()
#             cur_loss = loss.item()
#             clip_grad_value_(model.parameters(), clip_value = args.use_clip_norm_value)
#             optimizer.step()
#
#             # EVALUATION
#             model.eval()
#             perfs = []
#             for prefix in ["val", "test"]:
#                 pos_edge_index = data[f"{prefix}_pos_edge_index"]
#                 neg_edge_index = data[f"{prefix}_neg_edge_index"]
#                 with torch.no_grad():
#                     if args.use_moe:
#                         z, output1, output2, expert_list1, expert_list2 = model(data.x, data.train_pos_edge_index)
#                     elif args.expert_mhs_loss:
#                         z, output1, output2, expert_list1, expert_list2 = model(data.x, data.train_pos_edge_index)
#                     else:
#                         z = model(data.x, data.train_pos_edge_index)
#
#                     link_logits, edge_idx = model.decode(z, pos_edge_index, neg_edge_index)
#                 link_probs = link_logits.sigmoid()
#                 link_labels = get_link_labels(pos_edge_index, neg_edge_index)
#                 auc = roc_auc_score(link_labels.cpu(), link_probs.cpu())
#                 perfs.append(auc)
#
#             val_perf, tmp_test_perf = perfs
#             if val_perf > best_val_perf:
#                 best_val_perf = val_perf
#                 test_perf = tmp_test_perf
#             if epoch % 10 == 0:
#                 log = "Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}"
#                 print(log.format(epoch, cur_loss, best_val_perf, test_perf))
#
#         # FAIRNESS
#         auc = test_perf
#         cut = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
#         best_acc = 0
#         best_cut = 0.5
#         for i in cut:
#             acc = accuracy_score(link_labels.cpu(), link_probs.cpu() >= i)
#             if acc > best_acc:
#                 best_acc = acc
#                 best_cut = i
#         f = prediction_fairness(edge_idx.cpu(), link_labels.cpu(), link_probs.cpu() >= best_cut, Y.cpu())
#         acc_auc.append([best_acc * 100, auc * 100])
#         fairness.append([x * 100 for x in f])
#
#     # 3.evaluation results
#     ma = np.mean(np.asarray(acc_auc), axis=0)
#     mf = np.mean(np.asarray(fairness), axis=0)
#     sa = np.std(np.asarray(acc_auc), axis=0)
#     sf = np.std(np.asarray(fairness), axis=0)
#     # 3.1performance results
#     print(f"ACC: {ma[0]:2f} +- {sa[0]:2f}")
#     print(f"AUC: {ma[1]:2f} +- {sa[1]:2f}")
#     # 3.2fairness results
#     print(f"DP mix: {mf[0]:2f} +- {sf[0]:2f}")
#     print(f"EoP mix: {mf[1]:2f} +- {sf[1]:2f}")
#     print(f"DP group: {mf[2]:2f} +- {sf[2]:2f}")
#     print(f"EoP group: {mf[3]:2f} +- {sf[3]:2f}")
#     print(f"DP sub: {mf[4]:2f} +- {sf[4]:2f}")
#     print(f"EoP sub: {mf[5]:2f} +- {sf[5]:2f}")
#     print("args: ", args)
