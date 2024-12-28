import numpy as np
import torch
from torch_geometric.data import Data


def fully_connected_graph_builder(loss, std_H, K, graph_embedding_size):
    x1 = np.expand_dims(np.diag(std_H), axis=1)
    x2 = np.zeros((K, graph_embedding_size))
    x = np.concatenate((x1, x2), axis=1)
    x = torch.tensor(x, dtype=torch.float)

    # conisder fully connected graph
    loss2 = np.copy(loss)
    mask = np.eye(K)
    diag_loss2 = np.multiply(mask, loss2)
    loss2 = loss2 - diag_loss2
    attr_ind = np.nonzero(loss2)
    edge_attr = loss[attr_ind]
    edge_attr = np.expand_dims(edge_attr, axis=-1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    attr_ind = np.array(attr_ind)
    adj = np.zeros(attr_ind.shape)
    adj[0, :] = attr_ind[1, :]
    adj[1, :] = attr_ind[0, :]
    edge_index = torch.tensor(adj, dtype=torch.long)

    y = torch.tensor(np.expand_dims(loss, axis=0), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.contiguous(), edge_attr=edge_attr, y=y)
    return data

def graph_builder_partially_connected():
    pass

def proc_data_centralized(HH_real, H_std, K, init_emb_size):
    n = HH_real.shape[0]  # 即 layouts,每个layouts构建一张图
    data_list = []
    for i in range(n):
        data = fully_connected_graph_builder(HH_real[i,:,:], H_std[i,:,:], K, init_emb_size)
        data_list.append(data)
    return data_list



