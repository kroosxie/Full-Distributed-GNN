import numpy as np
import D2D_generator as D2D
import utils
import Graph_builder as Gbld
import torch
from torch_geometric.data import DataLoader

# 真实版信道生成
class init_parameters():
    def __init__(self):
        # wireless network settings
        self.n_links = D2DNum_K  # 这里是D2D直连链路数目
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

class AirMPNN(torch.nn.Module):
    def __init__(self):
        super(AirMPNN, self).__init__()

    def foward(self, data):
        pass

class LocalMPNN(torch.nn.Module):
    def __init__(self):
        super(AirMPNN, self).__init__()

    def foward(self, data):
        pass


D2DNum_K = 40
train_layouts = 2  # 模拟拓扑结构变化
train_timesteps = 3
train_config = init_parameters()

# Train data generation
print('Train data generation')
train_channel_real = D2D.train_channel_generator_1(train_config, train_layouts, train_timesteps) # 真实信道
train_channel_simplified = D2D.train_channel_generator_2(train_layouts, train_timesteps, D2DNum_K) # 简易信道（仿真使用）
train_directlink_losses = utils.get_directlink_losses(train_channel_simplified)

# Graph data processing
print('Graph data processing')
global_graph_fully_connected = Gbld.graph_builder_fully_connected()  # 构建全局图拓扑结构(全连接型)
global_graph_partially_connected = Gbld.graph_builder_partially_connected()  # 构建全局图拓扑结构（部分连接，即干扰链路小的边忽略不计）
local_graph_list = Gbld.collect_subgraph()  # 挑选并构建本地的子图（一跳或多跳）,并构成一个list

# Local training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_basic = LocalMPNN().to(device)  # 基础本地模型
model_over_the_air = AirMPNN().to(device)  # 后续可以加入over the air模块

optimizer = torch.optim.Adam(model_basic.parameters(), lr=0.002)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
train_loader = DataLoader(local_graph_list, batch_size=50, shuffle=True, num_workers=0)