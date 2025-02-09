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
# pro版可能更能采集直连链路特征
def normalize_data_pro(squared_train_data, K):
    # normalize train directlink
    mask = np.eye(K) 
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

def compute_SINRs(general_para, allocs, directlink_channel_losses, crosslink_channel_losses):
    # assert np.shape(directlink_channel_losses) == np.shape(allocs), \
    #     "Mismatch shapes: {} VS {}".format(np.shape(directlink_channel_losses), np.shape(allocs))
    SINRs_numerators = allocs * directlink_channel_losses
    SINRs_denominators = np.squeeze(np.matmul(crosslink_channel_losses, np.expand_dims(allocs, axis=-1))) + general_para.output_noise_power / general_para.tx_power  # layouts X N
    SINRs = SINRs_numerators / SINRs_denominators
    return SINRs

def compute_rates(general_para, allocs, channel_losses):
    directlink_channel_losses = np.diagonal(channel_losses, axis1=2, axis2=3)
    N = np.shape(channel_losses)[-1]
    crosslink_channel_losses = channel_losses * ((np.identity(N) < 1).astype(float))
    SINRs = compute_SINRs(general_para, allocs, directlink_channel_losses, crosslink_channel_losses)
    rates = np.log2(1 + SINRs)
    return rates

def batch_WMMSE(p_int, alpha, H, Pmax, var_noise):
    # 保存原始形状用于最后恢复维度
    original_shape = p_int.shape  # (test_layouts, frame_num, K, 1)

    # 展平前两个维度 (test_layouts, frame_num) -> N = test_layouts*frame_num
    N_total = original_shape[0] * original_shape[1]
    K = original_shape[2]

    # 将四维输入重整为三维 (N_total, K, 1)
    p_int_flat = p_int.reshape(N_total, K, 1)
    alpha_flat = alpha.reshape(N_total, K)
    H_flat = H.reshape(N_total, K, K)  # 假设输入H的原始形状为 (test_layouts, frame_num, K, K)

    # 后续保持原始算法逻辑不变
    N = p_int_flat.shape[0]
    b = np.sqrt(p_int_flat)
    f = np.zeros((N, K, 1))
    w = np.zeros((N, K, 1))

    mask = np.eye(K)

    # 关键维度转换点 (保持核心计算三维)
    rx_power = np.multiply(H_flat, b)
    rx_power_s = np.square(rx_power)
    valid_rx_power = np.sum(np.multiply(rx_power, mask), 1)

    interference = np.sum(rx_power_s, 2) + var_noise
    f = np.divide(valid_rx_power, interference)
    w = 1 / (1 - np.multiply(f, valid_rx_power))

    for ii in range(100):
        fp = np.expand_dims(f, 1)  # (N, 1, K, 1)
        rx_power = np.multiply(H_flat.transpose(0, 2, 1), fp)  # 保持三维转置
        valid_rx_power = np.sum(np.multiply(rx_power, mask), 1)

        bup = np.multiply(alpha_flat, np.multiply(w, valid_rx_power))
        rx_power_s = np.square(rx_power)

        wp = np.expand_dims(w, 1)
        alphap = np.expand_dims(alpha_flat, 1)
        bdown = np.sum(np.multiply(alphap, np.multiply(rx_power_s, wp)), 2)

        btmp = bup / bdown
        b = np.minimum(btmp, np.ones((N, K)) * np.sqrt(Pmax)) \
            + np.maximum(btmp, np.zeros((N, K))) - btmp

        bp = np.expand_dims(b, 1)
        rx_power = np.multiply(H_flat, bp)
        rx_power_s = np.square(rx_power)
        valid_rx_power = np.sum(np.multiply(rx_power, mask), 1)

        interference = np.sum(rx_power_s, 2) + var_noise
        f = np.divide(valid_rx_power, interference)
        w = 1 / (1 - np.multiply(f, valid_rx_power))

    # 恢复四维输出形状
    p_opt = np.square(b).reshape(original_shape)  # (test_layouts, frame_num, K, 1)
    return p_opt