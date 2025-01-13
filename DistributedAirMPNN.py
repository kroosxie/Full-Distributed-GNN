import numpy as np
import torch
from torch_geometric.nn.conv import MessagePassing
import D2D_generator as D2D
import utils
import Graph_builder as Gbld
from torch_geometric.data import DataLoader
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN


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
        self.tx_power = np.power(10, (self.tx_power_milli_decibel - 30) / 10)  # 将发射功率从分贝毫瓦（dBm）单位转换为瓦特（W）单位
        self.noise_density_milli_decibel = -169
        self.input_noise_power = np.power(10, ((self.noise_density_milli_decibel - 30) / 10)) * self.bandwidth
        self.output_noise_power = self.input_noise_power
        self.setting_str = "{} links in {}X{} field directLink_length:{}~{}".format(self.n_links, self.field_length, self.field_length,
                                                                self.shortest_directLink_length,
                                                                self.longest_directLink_length)

def sr_loss(train_data, out_p, K):
    power = out_p
    power = torch.reshape(power, (-1, K, 1))  # 形状为[K, K, 1]？
    abs_H_2 = train_data.y
    abs_H_2 = abs_H_2.permute(0, 2, 1)
    rx_power = torch.mul(abs_H_2, power)
    mask = torch.eye(K)
    mask = mask.to(device)
    valid_rx_power = torch.sum(torch.mul(rx_power, mask), 1)  # valid:有效信号
    interference = torch.sum(torch.mul(rx_power, 1-mask), 1) + var
    rate = rate = torch.log2(1 + torch.div(valid_rx_power, interference))
    sr = torch.mean(torch.sum(rate, 1))
    loss = torch.neg(sr)
    return loss



def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = sr_loss(data, out, train_K)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / train_layouts / frame_num
        

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())
        for i in range(1, len(channels))
    ])

class GConv(MessagePassing):
    def __init__(self, mlp1, mlp2, **kwargs):
        super(GConv, self).__init__(aggr='max', **kwargs)
        self.mlp1 = mlp1
        self.mlp2 = mlp2

    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=1)  # tmp：临时变量
        comb = self.mlp2(tmp)
        return torch.cat([x[:, :1], comb], dim=1)

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # PyTorch Geometric 中的 MessagePassing 类的一个方法，用于执行图卷积层中的信息传递过程。

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_j, edge_attr], dim=1)
        agg = self.mlp1(tmp)
        return agg

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.mlp1, self.mlp2)
    

class CentralizedMPNN(torch.nn.Module):
    def __init__(self):
        super(CentralizedMPNN, self).__init__()

        self.mlp1 = MLP([2 + graph_embedding_size, 32, 32])
        self.mlp2 = MLP([33 + graph_embedding_size, 16, graph_embedding_size])
        self.conv = GConv(self.mlp1, self.mlp2)  # 实例化
        self.h2o = MLP([graph_embedding_size, 16])
        self.h2o = Seq(*[self.h2o,Seq(Lin(16, 1, bias = True), Sigmoid())])

    def forward(self, data):
        x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        x1 = self.conv(x = x0, edge_index = edge_index, edge_attr = edge_attr)  # 实例化的具体应用,应该是共享参数
        x2 = self.conv(x = x1, edge_index = edge_index, edge_attr = edge_attr)
        out = self.conv(x = x2, edge_index = edge_index, edge_attr = edge_attr)
        output = self.h2o(out[:,1:])
        return output


train_K = 20
train_layouts = 100
frame_num = 10  # 每个layout下的帧数
graph_embedding_size = 8  # 节点初始为1+8=9维
train_config = init_parameters()
var = train_config.output_noise_power / train_config.tx_power
# 输出功率被归一化[0,1]，因此通过该功率计算sum_rate时，相应的噪声也应该归一化

# Train data generation
train_channel_losses = D2D.train_channel_loss_generator_1(train_config, train_layouts, frame_num)

# Data standardization/normalization
norm_train_loss = utils.normalize_train_data(train_channel_losses)
# Todo：目前使用channel loss，后续使用H吧
# 目前认为开方的目的是便于计算，因为规范化后数据的字面量就失去了物理意义，且便于后面的over-the-air的特征捕捉
# train_H = np.sqrt(train_channel_losses_merged)
# train_H_std = utils.standardize_centralized_train_data(train_H, train_K, train_layouts, frame_num)

# Graph data processing
train_loader_list = []
train_data_list = Gbld.proc_data_distributed_fc(train_channel_losses, norm_train_loss, train_K, graph_embedding_size)
for train_data in train_data_list:
    train_loader = DataLoader(train_data, batch_size=50, shuffle=True, num_workers=0)
    train_loader_list.append(train_loader)  # 该list为多frames的layouts组成的list，感觉不太对

# train of CentralizedMPNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CentralizedMPNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)  # 学习率调整

for epoch in range(1, 20):
    loss = train()
    scheduler.step()


print("end")

