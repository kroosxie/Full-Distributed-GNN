import numpy as np
import networkx as nx  # 用于可视化
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt

def standardize_train_data():
    pass

def normalize_train_data(train_data):
    train_mean = np.mean(train_data)
    train_std = np.std(train_data)
    norm_train = (train_data-train_mean)/train_std
    return norm_train

# 训练数据标准化
# 分别对直连链路和干扰链路规范化
# def standardize_centralized_train_data(train_data, K_num, layout_num, frm_num):
#     # standardize train directlink
#     mask = np.eye(K_num)  # 生成对角单位矩阵
#     train_copy = np.copy(train_data)  # 生成副本，注：python是引用传递
#     diag_data = np.multiply(mask, train_copy)
#     diag_mean = np.sum(diag_H) / layout_num / K_num / frm_num  # 计算直连链路H均值
#     # diag_var = np.sqrt(np.sum(np.square(diag_H)) / layout_num / K_num / frm_num)  # 源代码：计算标准差????此处存疑
#     diag_var = np.sqrt(np.sum(np.square(diag_H-diag_mean)) / layout_num / K_num / frm_num)  # xjc：应该这样？下面也要改
#     tmp_diag = (diag_H - diag_mean) / diag_var  # 标准化为正态分布
#     # standardize train interference link
#     off_diag = train_copy - diag_H
#     off_diag_mean = np.sum(off_diag) / layout_num / K_num / (K_num - 1) / frm_num
#     # off_diag_var = np.sqrt(np.sum(np.square(off_diag)) / layout_num / K_num / (K_num - 1) / frm_num)  # 源代码
#     off_diag_var = np.sqrt(np.sum(np.square(off_diag - off_diag_mean)) / layout_num / K_num / (K_num - 1) / frm_num)  # xjc：改动
#     tmp_off = (off_diag - off_diag_mean) / off_diag_var
#     tmp_off_diag = tmp_off - np.multiply(tmp_off, mask)
#     std_train = np.multiply(tmp_diag, mask) + tmp_off_diag  # 规范化后的训练集
#     return std_train

# 训练数据归一化

def normalize_centralized_train_data(squared_train_data, train_K):
    # normalize train directlink
    mask = np.eye(train_K) 
    train_copy = np.copy(squared_train_data) 
    diag_H = np.multiply(mask, train_copy)  
    diag_H_max = np.max(diag_H)
    diag_H_min = np.min(diag_H)
    tmp_diag = (diag_H - diag_H_min) / (diag_H_max - diag_H_min) 
    # normalize train interference link
    off_diag = train_copy - diag_H
    off_diag_max = np.max(off_diag)
    off_diag_min = np.min(off_diag)
    tmp_off = (off_diag - off_diag_min) / (off_diag_max - off_diag_min)
    tmp_off_diag = tmp_off - np.multiply(tmp_off, mask)

    norm_train = np.multiply(tmp_diag, mask) + tmp_off_diag 

    return norm_train

def get_directlink_losses(channel_losses):
    directlink_losses = []
    n = channel_losses.shape[0]
    m = channel_losses.shape[1]
    for i in range(n):
        directlink_losses.append(np.diagonal(channel_losses[i,:,:]))
    directlink_losses = np.array(directlink_losses)
    assert np.shape(directlink_losses)==(n, m)
    return directlink_losses

def graph_showing(data):
    '''
    args:
         data: torch_geometric.data.Data
    '''
    G = nx.Graph()
    edge_index = data['edge_index'].t()
    edge_index = np.array(edge_index.cpu())
    G.add_edges_from(edge_index)
    nx.draw_networkx(G)
    plt.show()