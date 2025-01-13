import numpy as np
import D2D_generator as D2D
import utils
import Graph_builder as Gbld
import torch
from torch_geometric.data import DataLoader
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN

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
        super(LocalConv, self).__init__(aggr='sum', **kwargs)  # 聚合方式设为sum
        self.local_model_list = local_model_list
        # Todo：将各节点的node_model取出，使用索引

    def message(self, x_i, x_j, edge_attr):
        # 生成消息
        # for each node:
        tmp = torch.cat([x_j, edge_attr], dim=1)
        # 将tmp tensor每个节点（i），即每一列的值通过for循环进入到local_model_list[i]中
        end_num = tmp.shape[0]
        for i in end_num:
            print("agg")

    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=1)
        # 聚合消息
        return tmp

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
        self.mlp_U = MLP([33 + graph_embedding_size, 16, graph_embedding_size])
        self.h2o = Seq(*[MLP([graph_embedding_size, 16]), Seq(Lin(16, 1, bias=True), Sigmoid())])

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
        
    # 提取子图，暂定一阶
    def collect_subgraph(self):
        pass

    def node_proc(self):
        pass

    # 可通过计算图计算梯度
    # 即用各自节点上的子图的loss使用.backward()
    # 可将邻居节点上的输出的required_grad设为False，在计算图隔绝邻居节点参数，以达到只计算自身节点上的参数梯度
    # 可查看计算图
    def computeLocalGrad(self):
        pass

    def forward(self, data):
        x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        x1 = self.localconv(x=x0, edge_index=edge_index, edge_attr=edge_attr)  # 第一帧
        x2 = self.localconv(x=x1, edge_index=edge_index, edge_attr=edge_attr)
        out = self.localconv(x=x2, edge_index=edge_index, edge_attr=edge_attr)
        output = self.h2o(out[:, 1:])
        grad = self.computeLocalGrad()

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)


train_K = 20
train_layouts = 100  # 模拟拓扑结构变化
frame_num = 10  # 每个layout下的帧数
graph_embedding_size = 8  # 节点初始为1+8=9维
train_config = init_parameters()
var = train_config.output_noise_power / train_config.tx_power

# Train data generation
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
train_loader = DataLoader(train_data_list, batch_size=1, shuffle=True, num_workers=0)

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

optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

for epch in range(1, 20):
    loss = train()
    scheduler.step()