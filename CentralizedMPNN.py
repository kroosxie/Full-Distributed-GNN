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
    # 将训练时以batch训练的形状[batchsize*K，1]拆分为[batchsize, K, 1]
    power = torch.reshape(power, (-1, K, 1))  # 形状为[batchsize, K, 1]
    abs_H_2 = train_data.y
    abs_H_2 = abs_H_2.permute(0, 2, 1)  # 注意这步，将信道矩阵行和列翻转，对应
    rx_power = torch.mul(abs_H_2, power)  # 对应元素相乘
    mask = torch.eye(K)
    mask = mask.to(device)
    valid_rx_power = torch.sum(torch.mul(rx_power, mask), 1)  # valid:有效信号
    interference = torch.sum(torch.mul(rx_power, 1-mask), 1) + var
    rate = torch.log2(1 + torch.div(valid_rx_power, interference))
    sr = torch.mean(torch.sum(rate, 1))
    loss = torch.neg(sr)
    return loss

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)  # 由于dataloader中打乱了顺序，那么输出的p是不是也是乱序？是否会带来错误？
        loss = sr_loss(data, out, train_K)  # data是打乱的话，out也是相应打乱后的顺序
        loss.backward()
        total_loss += loss.item() * data.num_graphs  # 为什么要乘？因为在sr_rate中有mean，所以可以直接乘
        optimizer.step()
    return total_loss / train_layouts / frame_num

def test():
    model.eval()
    total_loss = 0
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            print('out_p', out.t())
            loss = sr_loss(data, out, test_K)
            total_loss += loss.item() * data.num_graphs
    return total_loss / test_layouts / frame_num

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())
        for i in range(1, len(channels))
    ])

class GConv(MessagePassing):
    def __init__(self, mlp1, mlp2, **kwargs):
        super(GConv, self).__init__(aggr='max', **kwargs)  # 注意aggr为max
        self.mlp1 = mlp1  # 消息生成MLP
        self.mlp2 = mlp2  # 消息更新MLP

    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=1)  # tmp：临时变量
        comb = self.mlp2(tmp)
        return torch.cat([x[:, :1], comb], dim=1)  # cat是方便匹配layers（9）和h2o（8）的输入维度

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
        # 共享参数
        x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        x1 = self.conv(x = x0, edge_index = edge_index, edge_attr = edge_attr)  # 实例化的具体应用,应该是共享参数
        x2 = self.conv(x = x1, edge_index = edge_index, edge_attr = edge_attr)
        out = self.conv(x = x2, edge_index = edge_index, edge_attr = edge_attr)
        output = self.h2o(out[:,1:])
        return output


train_K = 20
train_layouts = 100  # 模拟拓扑结构变化
frame_num = 10  # 每个layout下的帧数
graph_embedding_size = 8  # 节点初始为1+8=9维
train_config = init_parameters()
var = train_config.output_noise_power / train_config.tx_power  # 为什么要除最大发射功率？
# 猜想是因此输出功率被归一化[0,1]，因此通过该功率计算sum_rate时，相应的噪声也应该归一化

# Train data generation
train_channel_losses = D2D.train_channel_loss_generator_1(train_config, train_layouts, frame_num)
#Treat multiple frames as multiple samples for MPNN 
# 在MPNN中将多个帧视为多个采样,即每一个帧也看做是一个layout
train_channel_losses_merged = train_channel_losses.reshape(train_layouts*frame_num, train_K, train_K)

# Data standardization/normalization
# ？要不要开方 还需要检查一下，如果开方的话，节点和边的特征为H，不开方为pathloss
norm_train_loss = utils.normalize_train_data(train_channel_losses_merged)
# Todo：目前使用channel loss，后续使用H吧
# 目前认为开方的目的是便于计算，因为规范化后数据的字面量就失去了物理意义，且便于后面的over-the-air的特征捕捉
# train_H = np.sqrt(train_channel_losses_merged)
# train_H_std = utils.standardize_centralized_train_data(train_H, train_K, train_layouts, frame_num)

# Graph data processing
train_data_list = Gbld.proc_data_centralized(train_channel_losses_merged, norm_train_loss, train_K, graph_embedding_size)
train_loader = DataLoader(train_data_list, batch_size=50, shuffle=True, num_workers=0)  # shuffle：乱序
# train_loader = DataLoader(train_data_list, batch_size=50, shuffle=False, num_workers=0)  # 不打乱顺序
# Todo：PYG的DataLoader数据打乱影不影响拓扑关系？
# 以及打乱的是list中的元素还是展开后的所有元素？
# 打乱的是list中元素的进行前向传播的顺序，相当于leyouts集不变，每个epoch按不同的顺序前向传播

# train of CentralizedMPNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CentralizedMPNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)  # 学习率调整

for epoch in range(1, 20):
    loss = train()
    print(f'epoch: {epoch}  train_loss: {loss}')
    scheduler.step()  # 动态调整优化器学习率


test_config = init_parameters()
test_layouts = 50
test_K = train_K
test_channel_losses = D2D.train_channel_loss_generator_1(test_config, test_layouts, frame_num) # 真实信道
test_channel_losses_merged = test_channel_losses.reshape(test_layouts*frame_num, test_K, test_K)
norm_test_loss = utils.normalize_train_data(test_channel_losses_merged)
# norm_test_loss = utils.normalize_data_pro(test_channel_losses, test_K)
test_data_list = Gbld.proc_data_centralized(test_channel_losses_merged, norm_test_loss, test_K, graph_embedding_size)
test_loader = DataLoader(test_data_list, batch_size=50, shuffle=False, num_workers=0)


#test for MPNN
sum_rate_mpnn = test()
print('MPNN average sum rate:', -sum_rate_mpnn)

#test for epa
Pepa = np.ones((test_layouts, frame_num, test_K))
rates_epa = utils.compute_rates(test_config, Pepa, test_channel_losses)
sum_rate_epa = np.mean(np.sum(rates_epa, axis=2))
print('EPA average sum rate (test):', sum_rate_epa)

#test for random
Prand = np.random.rand(test_layouts, frame_num, test_K)
rates_rand = utils.compute_rates(test_config, Prand, test_channel_losses)
sum_rate_rand = np.mean(np.sum(rates_rand, axis=2))
print('RandP average sum rate:', sum_rate_rand)

#test for wmmse
Pini = np.random.rand(test_layouts, frame_num, test_K, 1)
Y1 = utils.batch_WMMSE(Pini, np.ones([test_layouts*frame_num, test_K]),np.sqrt(test_channel_losses),1,var)
Y2 = Y1.reshape(test_layouts, frame_num, test_K)
rates_wmmse = utils.compute_rates(test_config, Y2, test_channel_losses)
sum_rate_wmmse = np.mean(np.sum(rates_wmmse, axis=2))
print('WMMSE average sum rate:', sum_rate_wmmse)


print("end")

