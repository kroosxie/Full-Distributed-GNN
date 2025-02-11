import numpy as np
import D2D_generator as D2D
import utils
import Graph_builder as Gbld
import torch
from torch_geometric.data import DataLoader
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from torchviz import make_dot

# 真实版信道生成
class init_parameters():
    def __init__(self):
        # wireless network settings
        self.n_links = train_K  # D2D direct link
        self.field_length = 400
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

# 考虑实际训练的噪声？
class LocalConv(MessagePassing):  # per layer
    def __init__(self, local_model_list, **kwargs):
        super(LocalConv, self).__init__(**kwargs)
        self.local_model_list = local_model_list

    def message(self, x_i, x_j, edge_attr, edge_index): # x_i为所有边的源节点，x_j为所有边的目标节点
        node_num = len(self.local_model_list)
        tmp = torch.cat([x_j, edge_attr], dim=1)
        neigh_tmp_list = [[] for _ in range(node_num)]
        node_msg_list = [[] for _ in range(node_num)]
        for src, neigh_tmp in zip(edge_index[0], tmp):
            neigh_tmp_list[src.item()].append(neigh_tmp) # 将message添加到对应源节点的列表中
        for node_idx in range(node_num):
            if len(neigh_tmp_list[node_idx]) == 0:
                neigh_tmp_list[node_idx] = torch.zeros((1, tmp.shape[1]), device=x_i.device) # 若无邻居节点，生成全零张量
            else:
                neigh_tmp_list[node_idx] = torch.stack(neigh_tmp_list[node_idx])
        # 计算每个节点的消息
        for node_idx in range(node_num):
            node_msg_list[node_idx] = self.local_model_list[node_idx].mlp_m(neigh_tmp_list[node_idx])
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

# 每个节点上都有自身的MLP_M和MLP_U，假设共享参数
# 若共享参数的话，意味着各层共享一个MLP_m和MLP_u
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


# class DistributedMPNN():
class DistributedMPNN(torch.nn.Module):  # per round
    def __init__(self, node_num):
        super(DistributedMPNN, self).__init__()
        self.node_model_list = torch.nn.ModuleList()
        for i in range(node_num):
            node_model = LocalModel().to(device)  # per node
            node_model.node_id = i
            self.node_model_list.append(node_model)
        self.localconv = LocalConv(self.node_model_list).to(device)
        self.localh2o = LocalH2O(self.node_model_list).to(device)
        # 创建优化器列表，每个 node_model 一个优化器
        # self.optimizers = [torch.optim.SGD(node_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4) for node_model in self.node_model_list]
        self.optimizers = [torch.optim.Adam(node_model.parameters(), lr=0.002) for node_model in self.node_model_list]
        self.schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9) for optimizer in self.optimizers] # 学习率调整

    # 提取子图，暂定一阶
    def collect_subgraph(self, data, power):
        # 初始化一个列表，用于存储每个节点的一阶子图
        subgraph_list = []
        for node_idx in range(data.num_nodes):
            # 抽取一阶子图，mapping是子图节点在原图中的索引, sub_node是子图节点在重新编号后的索引,
            # edge_mask是属于子图的边的mask，用于筛选哪些边属于子图
            sub_nodes, sub_edges, mapping, edge_mask = k_hop_subgraph(
                node_idx=node_idx,
                num_hops=1,
                edge_index=data.edge_index,
                relabel_nodes=True,  # 重新编号节点
                num_nodes=data.num_nodes,
                flow='target_to_source',  # 边的方向
            )
            # 筛选出入边
            in_edge_mask = sub_edges[1] == mapping[0]
            # 创建子图的 Data 对象
            subgraph = Data(
                x=data.x[sub_nodes],
                y=data.y[:, mapping, sub_nodes],
                edge_index=sub_edges[:, in_edge_mask],
                edge_attr=data.edge_attr[edge_mask][in_edge_mask],
                mapping=mapping,
                p=power[sub_nodes]
            )
            subgraph_list.append(subgraph)
        return subgraph_list

    # node achievable rate
    # sg:subgraph 可暂定为一阶子图
    # 注意用真实值计算，即data的y标签 所以使用norm数据训练是合理的
    def subgraph_rate(self, subgraph_list):  # 其实是node_rate而非subgraph的sum_rate
        subG_rate_list = []
        for subgraph in subgraph_list:
            power = subgraph.p
            abs_H_2 = subgraph.y
            abs_H_2 = abs_H_2.t()  # 为什么要转置？
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

    def computeFrameRate(self, abs_H_2 , out_p):
        K = out_p.shape[0]
        abs_H_2 = abs_H_2.squeeze(dim=0).t() # 注意需要转置对应
        rx_power = torch.mul(abs_H_2, out_p)
        mask = torch.eye(K)
        valid_rx_power = torch.sum(torch.mul(rx_power, mask), 1)  # valid:有效信号
        interference = torch.sum(torch.mul(rx_power, 1 - mask), 1) + var
        rate = torch.log2(1 + torch.div(valid_rx_power, interference))
        fr = torch.sum(rate, 0)
        return fr

    def average_neighbor_gradients(self, node_grad_list, edge_index):
        node_num = len(node_grad_list)
        # 将 edge_index 转换为邻居字典
        neighbors = {i: [] for i in range(node_num)}
        for src, dst in edge_index.t().tolist():
            neighbors[src].append(dst)
        averaged_grad_list = []
        for node_idx in range(node_num):
            # 获取当前节点的梯度
            current_grads = node_grad_list[node_idx]
            # 获取邻居节点的梯度
            neighbor_grads = []
            for neighbor_idx in neighbors[node_idx]:
                neighbor_grads.append(node_grad_list[neighbor_idx])

            # 计算邻居梯度的平均值
            if neighbor_grads:
                # 对每个参数计算平均梯度
                avg_grads = []
                for param_idx in range(len(current_grads)):
                    # 将所有邻居节点的对应参数梯度堆叠起来
                    stacked_grads = torch.stack([grads[param_idx] for grads in neighbor_grads])
                    # 计算平均值
                    avg_grads.append(torch.mean(stacked_grads, dim=0))
                averaged_grad_list.append(avg_grads)
            else:
                # 如果没有邻居节点，使用自身的梯度
                averaged_grad_list.append(current_grads)
        return averaged_grad_list

    def averaged_nodes_params(self):
        # 检查输入是否为空
        if not self.node_model_list:
            raise ValueError("node_model_list is NULL")
        avg_nodes_params = []
        for param_idx in range(len(list(self.node_model_list[0].parameters()))):
            stacked_params = torch.stack([list(node_model.parameters())[param_idx] for node_model in self.node_model_list])
            avg_nodes_params.append(torch.mean(stacked_params, dim=0))
        return avg_nodes_params

    def airAvg_neighbor_gradients(self, node_grad_list, edge_index):
        pass
    
    def average_node_gradients(self, frame_grad_list):
        node_num = len(frame_grad_list[0])
        node_avg_grad_list = []
        for node_idx in range(node_num):
            node_avg_grad = []
            node_grad_list = []
            for frame_grad in frame_grad_list:
                node_grad_list.append(frame_grad[node_idx])
            for grad_param_idx in range(len(node_grad_list[0])):
                stacked_grads = torch.stack([node_grad[grad_param_idx] for node_grad in node_grad_list])
                node_avg_grad.append(torch.mean(stacked_grads, dim=0))
            node_avg_grad_list.append(node_avg_grad)
        return node_avg_grad_list


    def update_gradients(self, averaged_grad_list):
        for node_idx, node_model in enumerate(self.node_model_list):
            for param, avg_grad in zip(node_model.parameters(), averaged_grad_list[node_idx]):
                if param.grad is not None:
                    param.grad.data = avg_grad.data  # 更新梯度

    def update_params(self, averaged_params):
        for model in self.node_model_list:
            for i, param in enumerate(model.parameters()):
                param.data = averaged_params[i]

    def forward(self, data_list):  # per layout
        layout_rate = 0
        frame_grad_list = []
        for data in data_list:  # per frame
            sum_local_loss = 0  # 本地loss之和，非真实速率
            # utils.graph_showing(data)
            x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
            x1 = self.localconv(x=x0, edge_index=edge_index, edge_attr=edge_attr)  # 第一个slot进行，对应Layer_1
            x2 = self.localconv(x=x1, edge_index=edge_index, edge_attr=edge_attr)
            out = self.localconv(x=x2, edge_index=edge_index, edge_attr=edge_attr)
            # 该p_out可用于训练过程中真实调整发射功率，也可仅用于传值计算loss
            output = self.localh2o(out[:, 1:])
            print('power:', output.t())
            local_rate_list = self.computeLocalRate(data, output)  # 由于子图计算，与真实存在偏差
            # 假设一个frame中收发对个数保持不变
            # if model.training:
            if self.training:
                # 在每输入一个frame的data前先清除模型梯度，防止梯度积累
                for node_model in self.node_model_list:
                    node_model.zero_grad()
                # 对每个节点的 local_rate 单独进行反向传播
                for local_rate in local_rate_list:
                    local_rate.backward(retain_graph=True)  # 注意每调用一次，梯度会累加，此处选用手动清零，也可用累加结果求平均
                    node_grad_list = []
                    for node_model in self.node_model_list:
                        node_grad_list.append([param.grad.clone() for param in node_model.parameters()])
                frame_grad_list.append(node_grad_list)
            frame_rate = self.computeFrameRate(data.y, output)  # 真实速率
            print('frame rate: ', frame_rate)
            layout_rate += frame_rate.item()
            local_loss = torch.stack(local_rate_list)
            sum_loss = local_loss.sum()  # 注：若部分连接，会忽略部分干扰，导致sum_rate偏大
            sum_local_loss += sum_loss.item()
            print('The sum of the local losses of the frame: ', sum_local_loss)
        if self.training:
            # 平均各layout中各节点自身frames的梯度，然后将模型参数在节点间统一
            layout_avg_grad = self.average_node_gradients(frame_grad_list)
            self.update_gradients(layout_avg_grad)  # 使用平均梯度更新模型参数
            # 更新模型参数
            for optimizer in self.optimizers:
                optimizer.step()
        return layout_rate

# 针对梯度计算，可分为per_frame或per_layout
# 两者类比于 SGD 和 mini-batch GD
# 暂定per_layout
def train():
    model.train()
    total_rate = 0
    iteration = 0
    for layout_data in train_loader:  # per layout
        iteration += 1
        layout_data = [frame_data.to(device) for frame_data in layout_data]
        # for optimizer in model.optimizers:
        #     optimizer.zero_grad()
        layout_sum_rate = model(layout_data)  # 获取真实和速率和每个节点的损失列表

        # 每一个layout的frame更新一次
        # 感觉这样更新梯度过于频繁，可能会失效
        # 计算邻居节点的平均梯度
        # 只有每个layout第一帧的拓扑结构即layout_data[0].edge_index，不合理
        # node_avg_grad_list = model.average_neighbor_gradients(node_grad_list, layout_data[0].edge_index)  # 为什么是layout_data[0]
        # node_avg_grad_list = model.airAvg_neighbor_gradients(node_grad_list, layout_data[0].edge_index, layout_data[0].y)  # over the air,相当于梯度加权

        # 放这里会不会是学习率衰减太快了
        if iteration % lr_update_interval == 0:
            for scheduler in model.schedulers:
                scheduler.step()

        # Todo:模型参数平均，类似与联邦学习
        # 文章中是全部节点的平均，这其实有点不够分布式
        if iteration % param_avg_interval == 0:
            avg_param = model.averaged_nodes_params()
            model.update_params(avg_param)  # 已验证成功更新

        total_rate += layout_sum_rate
    train_rate = total_rate/train_layouts/frame_num
    return train_rate  # 真实rate，非用于反向传播的loss

def test():
    model.eval()  # 将模型设置为评估模式：即固定参数
    total_rate = 0
    for layout_data in test_loader:
        layout_data = [frame_data.to(device) for frame_data in layout_data]
        with torch.no_grad():
            layout_sum_rate = model(layout_data)
            total_rate += layout_sum_rate
    test_rate = total_rate / test_layouts / frame_num
    return test_rate  # 真实rate


train_K = 20
train_layouts = 100  # 模拟拓扑结构变化
frame_num = 10  # 每个layout下的帧数
graph_embedding_size = 8  # 节点初始为1+8=9维
train_config = init_parameters()
var = train_config.output_noise_power / train_config.tx_power # 噪声归一化
param_avg_interval= 20  # 邻居节点平均模型参数的layout个数
lr_update_interval = 20  # 学习率更新相隔的layout个数
threshold_rate = 0.8  # 构建部分连接图拓扑时的边阈值权重，滤除干扰值小于加权平均值的干扰边，参考值：0.8 2 3


# Train data generation
'''
# Train data模拟根据实际环境进行实时训练，[layouts, frames, K, K]
# Train data分为多个layout，各layout间模拟节点移动，地理位置不同
# layout中分为多个frame，各frame间地理位置相同，快衰落不同，但有关联。
# 建立图拓扑结构时，使用信道loss阈值。好处是计算节点损失函数时意义明确，但拓扑信息存在变动。
# 也可考虑固定每个layout的图拓扑结构，可保证同一个layout下的图拓扑一致，且data-loader可以使用batch-size，但使用一阶子图计算时干扰不够准确；
# 目前暂定使用的是channel loss阈值
'''
train_channel_losses = D2D.train_channel_loss_generator_1(train_config, train_layouts, frame_num) # 真实信道
# train_losses_simplified = D2D.train_channel_loss_generator_2(train_layouts, train_frames, D2DNum_K) # 简易信道（仿真使用）
# train_directlink_losses = utils.get_directlink_losses(train_losses_simplified)

# Data standardization/normalization
norm_train_loss = utils.normalize_train_data(train_channel_losses)
# norm_train_loss = utils.normalize_data_pro(train_channel_losses, train_K)  # 对直连和干扰信道分别norm
#

# Graph data processing
print('Graph data processing')
# 构建部分连接图，即干扰链路小于阈值的边忽略不计
train_data_list = Gbld.proc_data_distributed_pc(train_channel_losses, norm_train_loss, train_K, graph_embedding_size,
                                                threshold_rate)
# batchsize暂设为1，简化逻辑，适应动态图
# batchsize=1有必要吗？需要进一步看看channel生成过程，确定layouts，frames之间的关系
# 后续可以再尝试调整
train_loader = DataLoader(train_data_list, batch_size=1, shuffle=False, num_workers=0)
# Todo：可尝试以多个layout为一个batch，注意不打乱顺序
# 没必要

# Local training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DistributedMPNN(train_K).to(device)  # 基础本地模型
# model_over_the_air = AirMPNN().to(device)  # 后续可以加入over the air模块

# Test data generation
test_config = init_parameters()
test_K = train_K
test_layouts = 50
test_channel_losses = D2D.train_channel_loss_generator_1(test_config, test_layouts, frame_num) # 真实信道
norm_test_loss = utils.normalize_train_data(test_channel_losses)
# norm_test_loss = utils.normalize_data_pro(test_channel_losses, test_K)
test_data_list = Gbld.proc_data_distributed_pc(test_channel_losses, norm_test_loss, test_K, graph_embedding_size, threshold_rate)
test_loader = DataLoader(test_data_list, batch_size=1, shuffle=False, num_workers=0)


#test for epa(train)
Pepa = np.ones((train_layouts, frame_num, train_K))
rates_epa = utils.compute_rates(train_config, Pepa, train_channel_losses)
sum_rate_epa = np.mean(np.sum(rates_epa, axis=2))
print('EPA average sum rate:', sum_rate_epa)

#test for random(train)
Prand = np.random.rand(train_layouts, frame_num, train_K)
rates_rand = utils.compute_rates(train_config, Prand, train_channel_losses)
sum_rate_rand = np.mean(np.sum(rates_rand, axis=2))
print('RandP average sum rate:', sum_rate_rand)


# for epoch in range(1, 3):  # 但实际情况中，应该没有epoch，每个数据都是实时的
#     loss_1 = train()
#     for scheduler in model.schedulers:
#         scheduler.step()
#     print(f"epoch: {epoch} train_rate: {loss_1}")

# 真实情况应该是有足够多的layout数目
# 若与集中式比较，将layouts数目设为集中式的train_layouts*epoch
# 若需要调整学习率，可在一定的一定数量的layouts后进行scheduler，这样的话是不是和batchsize能够进行融合？
loss_1 = train()
print(f"train_rate: {loss_1}")

loss_2 = test()
print(f"test_rate: {loss_2}")

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





