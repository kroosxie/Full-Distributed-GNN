import numpy as np
import torch
from torch_geometric.nn.conv import MessagePassing
import D2D_generator as D2D
import utils
import Graph_builder as Gbld
from torch_geometric.data import DataLoader
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN
from torchviz import make_dot

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
        self.setting_str = "{} links in {}X{} field directLink_length:{}~{}".format(self.n_links, self.field_length,
                                                                                    self.field_length,
                                                                                    self.shortest_directLink_length,
                                                                                    self.longest_directLink_length)


def sr_loss(train_data, out_p, K):
    power = out_p
    power = torch.reshape(power, (-1, K, 1))  # 形状为[batchsize, K, 1]
    abs_H_2 = train_data.y
    abs_H_2 = abs_H_2.permute(0, 2, 1)  # 注意这步，将信道矩阵行和列翻转，对应
    rx_power = torch.mul(abs_H_2, power)  # 对应元素相乘
    mask = torch.eye(K)
    mask = mask.to(device)
    valid_rx_power = torch.sum(torch.mul(rx_power, mask), 1)  # valid:有效信号
    interference = torch.sum(torch.mul(rx_power, 1 - mask), 1) + var
    rate = torch.log2(1 + torch.div(valid_rx_power, interference))
    sr = torch.mean(torch.sum(rate, 1))
    loss = torch.neg(sr)
    return loss

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())
        for i in range(1, len(channels))
    ])


# CGNN's structure
class GConv_parallel(MessagePassing):
    def __init__(self, mlp1, mlp2, **kwargs):
        super(GConv_parallel, self).__init__(aggr='max', **kwargs)  # 注意aggr为max
        self.mlp_m_list = torch.nn.ModuleList()
        self.mlp_u_list = torch.nn.ModuleList()
        for _ in range(train_K):
            # case1: 深度拷贝 mlp1 以生成独立副本_分布式
            # copied_mlp1 = copy.deepcopy(mlp1)
            # copied_mlp2 = copy.deepcopy(mlp2)
            # self.mlp_m_list.append(copied_mlp1)
            # self.mlp_u_list.append(copied_mlp1)

            # case2：直接添加原始 mlp1（共享参数）_集中式
            self.mlp_m_list.append(mlp1)
            self.mlp_u_list.append(mlp2)

            # case3：独立结构+参数绑定（共享参数）_集中式
            # self.shared_mlp_1 = mlp1
            # self.shared_mlp_2 = mlp1
            # copied_mlp1 = copy.deepcopy(mlp1)
            # copied_mlp2 = copy.deepcopy(mlp2)
            # for src_layer, dest_layer in zip(self.shared_mlp_1.children(), copied_mlp1.children()):
            #     if isinstance(src_layer, torch.nn.Linear):
            #         dest_layer.weight = src_layer.weight  # 权重共享
            #         dest_layer.bias = src_layer.bias  # 偏置共享
            # for src_layer, dest_layer in zip(self.shared_mlp_2.children(), copied_mlp2.children()):
            #     if isinstance(src_layer, torch.nn.Linear):
            #         dest_layer.weight = src_layer.weight  # 权重共享
            #         dest_layer.bias = src_layer.bias  # 偏置共享
            # self.mlp_m_list.append(copied_mlp1)
            # self.mlp_u_list.append(copied_mlp2)

    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=1)  # tmp：临时变量
        tmp_chunks = tmp.view(train_batchsize, train_K, -1)  # (batch_size, node_num, feature_dim)
        comb_list = [mlp_u(tmp_chunks[:, i, :]) for i, mlp_u in enumerate(self.mlp_u_list)]
        comb_stacked = torch.stack(comb_list, dim=1)
        comb = comb_stacked.flatten(0, 1)  # (batch_size*node_num, output_dim)
        return torch.cat([x[:, :1], comb], dim=1)  # cat是方便匹配layers（9）和h2o（8）的输入维度

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # PyTorch Geometric 中的 MessagePassing 类的一个方法，用于执行图卷积层中的信息传递过程。

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_j, edge_attr], dim=1)
        tmp_chunks = tmp.view(train_batchsize, train_K, train_K-1, -1)  # (batch_size, node_num, edge_num, feature_dim)
        agg_list = [mlp_m(tmp_chunks[:, i, :]) for i, mlp_m in enumerate(self.mlp_m_list)]
        agg_stacked = torch.stack(agg_list, dim=1)  # 形状变为 (batch_size, node_num, edge_num, feature_dim)
        agg = agg_stacked.flatten(0, 2)  # 展平前三维 → (batch_size * node_num * edge_num, feature_dim)
        return agg

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.mlp1, self.mlp2)

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
        # self.conv = GConv(self.mlp1, self.mlp2)  # 实例化
        self.conv = GConv_parallel(self.mlp1, self.mlp2)  # 并行计算
        self.h2o = Seq(*[MLP([graph_embedding_size, 16]), Seq(Lin(16, 1, bias=True), Sigmoid())])

    def forward(self, data):
        # 共享参数
        x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        x1 = self.conv(x=x0, edge_index=edge_index, edge_attr=edge_attr)  # 实例化的具体应用,应该是共享参数
        x2 = self.conv(x=x1, edge_index=edge_index, edge_attr=edge_attr)
        out = self.conv(x=x2, edge_index=edge_index, edge_attr=edge_attr)
        output = self.h2o(out[:, 1:])
        return output

def train_CGNN():
    model_CGNN.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer_CGNN.zero_grad()
        out = model_CGNN(data)  # 由于dataloader中打乱了顺序，那么输出的p是不是也是乱序？是否会带来错误？
        # print('train_out_p: ', out.t())
        loss = sr_loss(data, out, train_K)  # data是打乱的话，out也是相应打乱后的顺序
        # dot = make_dot(loss)
        # dot.render(filename="backward_graph", format="pdf")
        loss.backward()
        total_loss += loss.item() * data.num_graphs  # 为什么要乘？因为在sr_rate中有mean，所以可以直接乘
        optimizer_CGNN.step()
    return total_loss / train_layouts / frame_num

def test_CGNN():
    model_CGNN.eval()
    total_loss = 0
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            out = model_CGNN(data)
            print('out_p', out.t())
            loss = sr_loss(data, out, test_K)
            total_loss += loss.item() * data.num_graphs
    return total_loss / test_layouts / frame_num

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

class LocalModel(torch.nn.Module):
    def __init__(self):
        super(LocalModel, self).__init__()
        self.mlp_m = MLP([2 + graph_embedding_size, 32, 32])
        self.mlp_u = MLP([33 + graph_embedding_size, 16, graph_embedding_size])
        self.h2o = Seq(*[MLP([graph_embedding_size, 16]), Seq(Lin(16, 1, bias=True), Sigmoid())])

class DistributedMPNN(torch.nn.Module):  # per round
    def __init__(self, node_num):
        super(DistributedMPNN, self).__init__()
        self.node_model_list = torch.nn.ModuleList()
        for i in range(node_num):
            node_model = LocalModel().to(device)  # per node
            self.node_model_list.append(node_model)
        self.localconv = LocalConv(self.node_model_list).to(device)
        self.localh2o = LocalH2O(self.node_model_list).to(device)

    # 提取子图，暂定一阶
    def collect_subgraph(self, data, power):
        subgraph_list = []
        for node_idx in range(data.num_nodes):
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
            interference = rx_power.sum() - valid_rx_power + var
            # interference = interference.detach()  # 将interference设为环境变量，即requires_grad设为False
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

    def computeFrameRate(self, abs_H_2, out_p):
        K = out_p.shape[0]
        abs_H_2 = abs_H_2.squeeze(dim=0).t()  # 注意需要转置对应
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
            stacked_params = torch.stack(
                [list(node_model.parameters())[param_idx] for node_model in self.node_model_list])
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

    def sum_node_gradients(self, node_grad_list):
        node_sum_grad = []
        for grad_param_idx in range(len(node_grad_list[0])):
            stacked_grads = torch.stack([node_grad[grad_param_idx] for node_grad in node_grad_list])
            node_sum_grad.append(torch.sum(stacked_grads, dim=0))
        return node_sum_grad

    def avg_node_gradients(self, node_grad_list):
        node_avg_grad = []
        for grad_param_idx in range(len(node_grad_list[0])):
            stacked_grads = torch.stack([node_grad[grad_param_idx] for node_grad in node_grad_list])
            node_avg_grad.append(torch.mean(stacked_grads, dim=0))
        return node_avg_grad

    def update_gradients(self, averaged_grad_list):
        for node_idx, node_model in enumerate(self.node_model_list):
            for param, avg_grad in zip(node_model.parameters(), averaged_grad_list):
                if param.grad is not None:
                    param.grad.data = avg_grad.data  # 更新梯度

    def update_params(self, averaged_params):
        for model in self.node_model_list:
            for i, param in enumerate(model.parameters()):
                param.data = averaged_params[i]

    def forward(self, data_list):  # per batch/layout
        layout_rate = 0
        frame_grad_list = []
        for data in data_list:  # per frame
            sum_local_loss = 0  # 本地loss之和，非真实速率
            x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
            x1 = self.localconv(x=x0, edge_index=edge_index, edge_attr=edge_attr)  # 第一个slot进行，对应Layer_1
            x2 = self.localconv(x=x1, edge_index=edge_index, edge_attr=edge_attr)
            out = self.localconv(x=x2, edge_index=edge_index, edge_attr=edge_attr)
            output = self.localh2o(out[:, 1:])
            print('power:', output.t())
            local_rate_list = self.computeLocalRate(data, output)  # 若非全连接，则与真实速率存在偏差
            frame_rate = self.computeFrameRate(data.y, output)  # 真实速率
            print('frame rate: ', frame_rate)
            layout_rate += frame_rate.item()
            local_loss = torch.stack(local_rate_list)
            sum_loss = local_loss.sum()  # 注：若部分连接，会忽略部分干扰，导致sum_rate偏大
            sum_local_loss += sum_loss.item()
            print('The sum of the local losses of the frame: ', sum_local_loss)

            if self.training:
                for optimizer in self.optimizers:
                    optimizer.zero_grad()
                for local_rate in local_rate_list:
                    local_rate.backward(retain_graph=True)  # 注意每调用一次，梯度会累加，此处选用手动清零，也可用累加结果求平均
                node_grad_list_tmp = []
                for node_model in self.node_model_list:
                    node_grad_list_tmp.append([param.grad.clone() for param in node_model.parameters()])
                node_grad_param = self.sum_node_gradients(node_grad_list_tmp)  # 因为计算图的梯度累加，这个其实就是sum_rate针对的GNN统一参数的梯度
                frame_grad_list.append(node_grad_param)

        if self.training:
            # 平均各layout中各节点自身frames的梯度，然后将模型参数在节点间统一
            layout_avg_grad = self.avg_node_gradients(frame_grad_list)
            self.update_gradients(layout_avg_grad)  # 使用平均梯度更新模型参数
            for optimizer in optimizers_DGNN:
                optimizer.step()
            print('node param updated')
        return layout_rate

def train_DGNN():
    model.train()
    total_rate = 0
    iteration = 0
    # 初始参数平均
    avg_param = model.averaged_nodes_params()
    model.update_params(avg_param)
    print('node param initially averaged')
    for layout_data in train_loader:  # per layout
        iteration += 1
        print(f'------第{iteration}个layout------')
        layout_data = [frame_data.to(device) for frame_data in layout_data]
        layout_sum_rate = model(layout_data)  # 获取真实和速率和每个节点的损失列表
        # 学习率更新
        if iteration % lr_update_interval == 0:
            for scheduler in model.schedulers:
                scheduler.step()
            print('scheduler updated')
        total_rate += layout_sum_rate
    train_rate = total_rate/train_layouts/frame_num
    return train_rate  # 真实rate，非用于反向传播的loss

def test_DGNN():
    model_DGNN.eval()  # 将模型设置为评估模式：即固定参数
    total_rate = 0
    for layout_data in test_loader:
        layout_data = [frame_data.to(device) for frame_data in layout_data]
        with torch.no_grad():
            layout_sum_rate = model(layout_data)
            total_rate += layout_sum_rate
    test_rate = total_rate / test_layouts / frame_num
    return test_rate  # 真实rate

train_K = 20
train_layouts = 2000
frame_num = 10
graph_embedding_size = 8
train_config = init_parameters()
var = train_config.output_noise_power / train_config.tx_power
train_batchsize = 50

# Train data generation
train_channel_losses = D2D.train_channel_loss_generator_1(train_config, train_layouts, frame_num)
# Treat multiple frames as multiple samples for MPNN
train_channel_losses_merged = train_channel_losses.reshape(train_layouts * frame_num, train_K, train_K)

# Data standardization/normalization
norm_train_loss = utils.normalize_train_data(train_channel_losses_merged)

# Graph data processing
train_data_list = Gbld.proc_data_centralized(train_channel_losses_merged, norm_train_loss, train_K,
                                             graph_embedding_size)
train_loader = DataLoader(train_data_list, batch_size=train_batchsize, shuffle=True, num_workers=0)  # shuffle：乱序
# train_loader = DataLoader(train_data_list, batch_size=50, shuffle=False, num_workers=0)  # 不打乱顺序

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train of CentralizedMPNN
model_CGNN = CentralizedMPNN().to(device)
optimizer_CGNN = torch.optim.Adam(model_CGNN.parameters(), lr=0.002)
scheduler_CGNN = torch.optim.lr_scheduler.StepLR(optimizer_CGNN, step_size=20, gamma=0.9)  # 学习率调整

for epoch in range(1, 20):
    loss_CGNN = train_CGNN()
    print(f'epoch: {epoch}  train_loss_CGNN: {loss_CGNN}')
    scheduler_CGNN.step()  # 动态调整优化器学习率

# save CGNN param
torch.save(model_CGNN.state_dict(), 'CGNN_param.pth')
# delete useless param
# keys_to_remove = [key for key in state_dict.keys() if key.startswith('conv')]
# for key in keys_to_remove:
#     del state_dict[key]

# train of DistributedMPNN
model_DGNN = DistributedMPNN(train_K).to(device)
for node_model in model_DGNN.node_model_list:
    node_model.load_state_dict(torch.load('CGNN_param.pth'))
optimizers_DGNN = [torch.optim.Adam(node_model.parameters(), lr) for node_model in model_DGNN.node_model_list]
schedulers_DGNN = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9) for optimizer in optimizers_DGNN]

for epoch in range(1, 20):
    loss_DGNN = train_DGNN()
    print(f'epoch: {epoch}  train_loss_DGNN: {loss_DGNN}')
    scheduler_DGNN.step()


test_config = init_parameters()
test_layouts = 50
test_K = train_K
test_channel_losses = D2D.train_channel_loss_generator_1(test_config, test_layouts, frame_num)  # 真实信道
test_channel_losses_merged = test_channel_losses.reshape(test_layouts * frame_num, test_K, test_K)
norm_test_loss = utils.normalize_train_data(test_channel_losses_merged)
# norm_test_loss = utils.normalize_data_pro(test_channel_losses, test_K)
test_data_list = Gbld.proc_data_centralized(test_channel_losses_merged, norm_test_loss, test_K, graph_embedding_size)
test_loader = DataLoader(test_data_list, batch_size=50, shuffle=False, num_workers=0)


# test for CentralizedMPNN
sum_rate_mpnn = test_CGNN()
print('CentralizedMPNN average sum rate:', -sum_rate_mpnn)

# test for DistributedMPNN
state_dict = torch.load('CGNN_param.pth')
# 定义键名映射关系
keys_to_remove = [key for key in state_dict.keys() if key.startswith('conv')]
for key in keys_to_remove:
    del state_dict[key]
key_mapping = {'mlp1': 'mlp_m', 'mlp2': 'mlp_u'}
modefied_state_dict = {}
for key, value in state_dict.items():
    for old_key, new_key in key_mapping.items():
        if old_key in key:
            key = key.replace(old_key, new_key)
            break
    modefied_state_dict[key] = value
for node_model in model.node_model_list:
    node_model.load_state_dict(modefied_state_dict)

sum_rate_mpnn = test_DGNN()
print('CentralizedMPNN average sum rate:', -sum_rate_mpnn)

# test for epa
Pepa = np.ones((test_layouts, frame_num, test_K))
rates_epa = utils.compute_rates(test_config, Pepa, test_channel_losses)
sum_rate_epa = np.mean(np.sum(rates_epa, axis=2))
print('EPA average sum rate (test):', sum_rate_epa)

# test for random
Prand = np.random.rand(test_layouts, frame_num, test_K)
rates_rand = utils.compute_rates(test_config, Prand, test_channel_losses)
sum_rate_rand = np.mean(np.sum(rates_rand, axis=2))
print('RandP average sum rate:', sum_rate_rand)

# test for wmmse
Pini = np.random.rand(test_layouts, frame_num, test_K, 1)
Y1 = utils.batch_WMMSE(Pini, np.ones([test_layouts * frame_num, test_K]), np.sqrt(test_channel_losses), 1, var)
Y2 = Y1.reshape(test_layouts, frame_num, test_K)
rates_wmmse = utils.compute_rates(test_config, Y2, test_channel_losses)
sum_rate_wmmse = np.mean(np.sum(rates_wmmse, axis=2))
print('WMMSE average sum rate:', sum_rate_wmmse)

print("end")

