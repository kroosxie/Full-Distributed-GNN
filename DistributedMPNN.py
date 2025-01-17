import numpy as np
import D2D_generator as D2D
import utils
import Graph_builder as Gbld
import torch
from torch_geometric.data import DataLoader
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_geometric.data import Data
import networkx as nx  # 用于可视化
import matplotlib.pyplot as plt

# 真实版信道生成
class init_parameters():
    def __init__(self):
        # wireless network settings
        self.n_links = train_K  # 这里是D2D直连链路数目
        self.field_length = 1000
        self.shortest_directLink_length = 2
        self.longest_directLink_length = 40
        self.shortest_crossLink_length = 1  # crosslink即干扰链路，设置最小干扰链路距离防止产生过大的干扰
        self.bandwidth = 5e6
        self.carrier_f = 2.4e9
        self.tx_height = 1.5
        self.rx_height = 1.5
        self.antenna_gain_decibel = 2.5
        self.tx_power_milli_decibel = 40  # 最大发射功率为40dBm
        self.tx_power = np.power(10, (self.tx_power_milli_decibel - 30) / 10)  # np.power为乘方
        self.noise_density_milli_decibel = -169
        self.input_noise_power = np.power(10, ((self.noise_density_milli_decibel - 30) / 10)) * self.bandwidth
        self.output_noise_power = self.input_noise_power
        self.setting_str = "{} links in {}X{} field directLink_length:{}~{}".format(self.n_links, self.field_length, self.field_length,
                                                                self.shortest_directLink_length,
                                                                self.longest_directLink_length)

# 整图输入，整图输出
# 该类实现MessagePassing的过程
# 使用一跳子图
# 考虑实际训练的噪声？
# 可能需要节点list
class LocalConv(MessagePassing):  # per layer
    def __init__(self, local_model_list, **kwargs):
        super(LocalConv, self).__init__(**kwargs)
        self.local_model_list = local_model_list

    def message(self, x_i, x_j, edge_attr, edge_index): # x_i为所有边的源节点，x_j为所有边的目标节点
        node_num = len(self.local_model_list)
        tmp = torch.cat([x_j, edge_attr], dim=1)
        edge_num = tmp.shape[0]
        neigh_tmp_list = [[] for _ in range(node_num)]
        node_msg_list = [[] for _ in range(node_num)]
        for src, neigh_tmp in zip(edge_index[0], tmp):
            # 将message添加到对应源节点的列表中
            neigh_tmp_list[src.item()].append(neigh_tmp)
        neigh_tmp_list = [torch.stack(neigh_tmps) if neigh_tmps else torch.tensor([]) for neigh_tmps in neigh_tmp_list]
        for node_idx in range(node_num):
            # 注意，无边节点的维度存在问题
            # Todo：设置0阈值或者小阈值或者全连接，避免节点无连接的边的情况，后续再进一步考虑无连接节点的情况
            node_msg_list[node_idx] = self.local_model_list[node_idx].mlp_m(neigh_tmp_list[node_idx])
            # neigh_msg_list.append(neigh_msg)  # 无边节点会出现问题
        return node_msg_list

    def aggregate(self, node_msg_list, aggr = 'sum'):  # 聚合方式暂设为sum，可与over-the-air适配
        node_agg_list = []
        aggr_funcs = {
            'sum': torch.sum,
            'mean': torch.mean,
            'max': torch.max
        }
        for node_idx, msgs in enumerate(node_msg_list):
            if msgs.nelement() == 0:  # 无边节点设为全0张量
                aggregated_msg = torch.zeros(self.local_model_list[node_idx].mlp_m.output_dim, device=msgs.device)
            else:
                aggregated_msg = aggr_funcs[aggr](msgs, dim=0)
            node_agg_list.append(aggregated_msg)
        aggregated_msgs = torch.stack(node_agg_list)  # 将list转化为tensor
        return aggregated_msgs

    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=1)
        node_num = len(self.local_model_list)
        comb_list = [[] for _ in range(node_num)]
        for node_idx in range(node_num):
            comb_list[node_idx] = self.local_model_list[node_idx].mlp_u(tmp[node_idx])
        comb = torch.stack(comb_list)
        return torch.cat([x[:, :1], comb], dim=1)

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())
        for i in range(1, len(channels))
    ])

# sg:subgraph 可暂定为一阶子图
# 注意用真实值计算，即data的y标签
# 所以使用norm数据训练是合理的
def local_loss(out_p_sg, x_v, edge_u):
    pass

# 每个节点上都有自身的MLP_M和MLP_U
# 若共享参数的话，意味着每一层共享一个MLP_m和MLP_u
class LocalModel(torch.nn.Module):
    def __init__(self):
        super(LocalModel, self).__init__()
        self.node_id = None  # 初始为None
        self.mlp_m = MLP([2 + graph_embedding_size, 32, 32])
        self.mlp_u = MLP([33 + graph_embedding_size, 16, graph_embedding_size])
        self.h2o = Seq(*[MLP([graph_embedding_size, 16]), Seq(Lin(16, 1, bias=True), Sigmoid())])

class LocalH2O(torch.nn.Module):  # 需要继承吗？
    def __init__(self, local_model_list, **kwargs):
        super(LocalH2O, self).__init__()
        self.local_model_list = local_model_list
        # self.local_h2o_list = [model.h2o for model in local_model_list]

    def forward(self, data_out):
        node_num = len(self.local_model_list)
        out_p_list = [[] for _ in range(node_num)]
        for node_idx in range(node_num):
            out_p_list[node_idx] = self.local_model_list[node_idx].h2o(data_out[node_idx])
        out_p = torch.stack(out_p_list)
        return out_p


class DistributedMPNN(torch.nn.Module):  # per round
    def __init__(self, node_num):
        super(DistributedMPNN, self).__init__()
        # self.node_model_list = []
        self.node_model_list = torch.nn.ModuleList()
        for i in range(node_num):
            node_model = LocalModel().to(device)  # per node
            node_model.node_id = i
            self.node_model_list.append(node_model)
        self.localconv = LocalConv(self.node_model_list).to(device)
        self.localh2o = LocalH2O(self.node_model_list).to(device)
        # 创建优化器列表，每个 node_model 一个优化器
        # self.optimizers = [torch.optim.SGD(node_model.parameters(), lr=0.01) for node_model in self.node_model_list]
        self.optimizers = [torch.optim.Adam(node_model.parameters(), lr=0.002) for node_model in self.node_model_list]

    # 提取子图，暂定一阶
    def collect_subgraph(self, data, power):
        # 初始化一个列表，用于存储每个节点的一阶子图
        subgraph_list = []
        for node_idx in range(data.num_nodes):
            # 抽取一阶子图
            sub_nodes, sub_edges, mapping, edge_mask = k_hop_subgraph(
                node_idx=node_idx,
                num_hops=1,
                edge_index=data.edge_index,
                relabel_nodes=True,  # 重新编号节点
                # flow='source_to_target',  # 边的方向
                flow='target_to_source',
            )

            # data = to_networkx(data)
            # nx.draw(data, with_labels=data.nodes)
            # plt.show()

            # 筛选出入边
            in_edge_mask = sub_edges[1] == mapping[0]
            in_sub_edges = sub_edges[:, in_edge_mask]
            in_edge_features = data.edge_attr[edge_mask][in_edge_mask]

            edge_mask_test = edge_mask[in_edge_mask],  # 子图的边在原图中的位置

            # 创建子图的 Data 对象
            subgraph = Data(
                x=data.x[sub_nodes],  # 子图的节点特征
                y=data.y[:,mapping, sub_nodes],  # 暂时只能用于全连接图，需要后续验证及改进
                edge_index=in_sub_edges,  # 子图的边连接信息
                edge_attr=in_edge_features,
                mapping=mapping,  # 目标节点在子图中的位置
                edge_mask=edge_mask[in_edge_mask],  # 子图的边在原图中的位置
                p=power[sub_nodes]
            )

            # 将子图添加到列表中
            subgraph_list.append(subgraph)
        return subgraph_list

    # node achievable rate
    def subgraph_rate(self, subgraph_list):
        subG_rate_list = []
        for subgraph in subgraph_list:
            power = subgraph.p
            abs_H_2 = subgraph.y
            abs_H_2 = abs_H_2.t()
            rx_power = torch.mul(power, abs_H_2)
            valid_rx_power = rx_power[subgraph.mapping]
            interference = rx_power.sum()-valid_rx_power + var
            # interference的requires_grad设为False 在此处对吗？
            interference = interference.detach()
            rate = torch.log2(1 + torch.div(valid_rx_power, interference))
            subG_loss = torch.neg(rate)
            subG_rate_list.append(subG_loss)
        return subG_rate_list

    # 可通过计算图计算梯度
    # 即用各自节点上的子图的loss使用.backward()
    # 可将邻居节点上的输出的required_grad设为False，在计算图隔绝邻居节点参数，以达到只计算自身节点上的参数梯度
    # 可查看计算图
    def computeLocalRate(self, data, out_p):
        subG_list = self.collect_subgraph(data, out_p)
        subG_rate_list = self.subgraph_rate(subG_list)
        return subG_rate_list


    def forward(self, data_list):
        for optimizer in self.optimizers: # 放这里合适吗？
            optimizer.zero_grad()
        for data in data_list:
            x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
            x1 = self.localconv(x=x0, edge_index=edge_index, edge_attr=edge_attr)  # 第一个slot进行，对应Layer_1
            x2 = self.localconv(x=x1, edge_index=edge_index, edge_attr=edge_attr)
            out = self.localconv(x=x2, edge_index=edge_index, edge_attr=edge_attr)
            # 该p_out可用于训练过程中真实调整发射功率，也可仅用于传值计算loss
            output = self.localh2o(out[:, 1:])
            local_rate_list = self.computeLocalRate(data, output)
            localRate = torch.stack(local_rate_list)
            sum_rate = localRate.sum()  # 注：若部分连接，会忽略部分干扰，导致sum_rate偏大
            for local_rate in local_rate_list:
                local_rate.backward(retain_graph=True)
        # grad_list = self.computeLocalGrad()  # 计算每个节点的本地梯度
        # grad_avg = np.mean(grad_list)
        return sum_rate

# 通过一阶子图，计算单个节点处的loss
def node_loss():
    pass

# 针对梯度计算，可分为per_frame或per_layout
# 两者类比于 SGD 和 mini-batch GD
# 暂定per_layout
def train():
    model.train()
    total_loss = 0
    # optimizers = [optim.SGD(mlp.parameters(), lr=0.01) for mlp in mlp_list]
    for layout_data in train_loader:
        for frame_data in layout_data:
            frame_data = frame_data.to(device)
            # p_out, grad_out = model(frame_data) # 该p_out可用于训练过程中真实调整发射功率，也可仅用于传值计算loss
        # optimizer.zero_grad()  # ? 需要吗
        p_out, grad_out = model(layout_data)



train_K = 20
train_layouts = 100  # 模拟拓扑结构变化
frame_num = 10  # 每个layout下的帧数
graph_embedding_size = 8  # 节点初始为1+8=9维
train_config = init_parameters()
var = train_config.output_noise_power / train_config.tx_power

# Train data generation
# layouts彼此间地理拓扑结构不同
# 一个layout中分为多个frames，frame间地理拓扑相同，快衰落不同，但有关联。
# 若建立子图时使用距离阈值，可保证同一个layout下的图拓扑一致，但使用一阶子图计算时可能忽略某些较大干扰信道；
# 若建立子图时使用信道loss阈值，计算节点损失函数时意义明确，但拓扑信息存在变动。
# 目前使用的是channel loss阈值
print('Train data generation')
train_channel_losses = D2D.train_channel_loss_generator_1(train_config, train_layouts, frame_num) # 真实信道
# train_losses_simplified = D2D.train_channel_loss_generator_2(train_layouts, train_frames, D2DNum_K) # 简易信道（仿真使用）
# train_directlink_losses = utils.get_directlink_losses(train_losses_simplified)

# Data standardization/normalization
norm_train_loss = utils.normalize_train_data(train_channel_losses)


# Graph data processing
print('Graph data processing')
# 构建部分连接图，即干扰链路小于阈值的边忽略不计
train_data_list = Gbld.proc_data_distributed_pc(train_channel_losses, norm_train_loss, train_K, graph_embedding_size)
# local_graph_list = Gbld.collect_subgraph(global_graph_partially_connected)  # 挑选并构建本地的子图（一跳或多跳）,并构成一个list
# batchsize暂设为1，简化逻辑，适应动态图
# batchsize=1有必要吗？需要进一步看看channel生成过程，确定layouts，frames之间的关系
# 后续可以再尝试调整
train_loader = DataLoader(train_data_list, batch_size=1, shuffle=False, num_workers=0)
# 可尝试以一个layout的多个帧为一个batch，注意不打乱顺序
# train_loader = DataLoader(train_data_list, batch_size=frame_num, shuffle=False, num_workers=0)

# Local training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = DistributedMPNN(train_K).to(device)  # 基础本地模型
model = DistributedMPNN(train_K)
# model_over_the_air = AirMPNN().to(device)  # 后续可以加入over the air模块

for name, param in model.named_parameters():
    print(f"Name: {name}")
    print(f"Type: {type(param)}")
    print(f"Size: {param.size()}")
    print(f"Values: {param}")

# optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

for epoch in range(1, 20):
    loss = train()
    scheduler.step()