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
    edge_attr = std_H[attr_ind]  # 修改错误
    edge_attr = np.expand_dims(edge_attr, axis=-1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    attr_ind = np.array(attr_ind)
    adj = np.zeros(attr_ind.shape)
    adj[0, :] = attr_ind[1, :]
    adj[1, :] = attr_ind[0, :]
    edge_index = torch.tensor(adj, dtype=torch.long)
    # Todo：将标签y也进行部分连接处理
    y = torch.tensor(np.expand_dims(loss, axis=0), dtype=torch.float)  # label y用于存储真实信道值
    data = Data(x=x, edge_index=edge_index.contiguous(), edge_attr=edge_attr, y=y)
    return data

def partially_connected_graph_builder(loss, std_H, K, graph_embedding_size):
    x1 = np.expand_dims(np.diag(std_H), axis=1)
    x2 = np.zeros((K, graph_embedding_size))
    x = np.concatenate((x1, x2), axis=1)
    x = torch.tensor(x, dtype=torch.float)

    # conisder fully connected graph
    loss2 = np.copy(loss)
    mask = np.eye(K)
    diag_loss2 = np.multiply(mask, loss2)
    loss2 = loss2 - diag_loss2
    attr_ind = np.nonzero(loss2)  # 返回非零值的行索引和列索引的tuple
    edge_attr = std_H[attr_ind]
    edge_attr = np.expand_dims(edge_attr, axis=-1)
    edge_attr_avg = np.mean(edge_attr)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # 滤除小于阈值的边
    # 如果出现某节点的所有的临边都被滤除的情况怎么办？
    threshold_rate = 0.8
    # threshold_rate = 3
    threshold = edge_attr_avg * threshold_rate
    mask_filter = edge_attr.squeeze() > threshold
    filtered_edge_attr = edge_attr[mask_filter]
    filtered_attr_ind = attr_ind[0][mask_filter.numpy()], attr_ind[1][mask_filter.numpy()]

    # 构建边索引
    adj = np.zeros((2, len(filtered_attr_ind[0])))
    adj[0, :] = filtered_attr_ind[1]
    adj[1, :] = filtered_attr_ind[0]  # 为什么要反着来？
    edge_index = torch.tensor(adj, dtype=torch.long)

    # 标签（全连接）
    y = torch.tensor(np.expand_dims(loss, axis=0), dtype=torch.float)  # label y用于存储真实信道值
    data = Data(x=x, edge_index=edge_index.contiguous(), edge_attr=filtered_edge_attr, y=y)
    return data


def proc_data_centralized(HH_real, H_std, K, init_emb_size):
    n = HH_real.shape[0]  # 即 layouts,每个合并layouts构建一张图
    data_list = []
    for i in range(n):
        data = fully_connected_graph_builder(HH_real[i,:,:], H_std[i,:,:], K, init_emb_size)
        data_list.append(data)
    return data_list

# fully connection
def proc_data_distributed_fc(HH_real, H_std, K, init_emb_size):
    n = HH_real.shape[0]  # 即 layouts,每个layouts构建一张图
    m = HH_real.shape[1]  # 每个frame也是一张图
    layouts_list = []
    data_list = []
    for i in range(n):
        for j in range(m):
            data = fully_connected_graph_builder(HH_real[i,j,:,:], H_std[i,j,:,:], K, init_emb_size)
            layouts_list.append((data))
        data_list.append(layouts_list)
    return data_list

# partially connection
def proc_data_distributed_pc(HH_real, H_std, K, init_emb_size):
    n = HH_real.shape[0]  # 即 layouts,每个layouts构建一张图
    m = HH_real.shape[1]  # 每个frame也是一张图
    frames_list = []
    layouts_list = []
    for i in range(n):
        for j in range(m):
            data = partially_connected_graph_builder(HH_real[i,j,:,:], H_std[i,j,:,:], K, init_emb_size)
            frames_list.append((data))
        layouts_list.append(frames_list)
        frames_list = []
    return layouts_list