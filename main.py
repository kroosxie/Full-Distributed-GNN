import numpy as np
import D2D_generator_1 as D2D
import utils

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

D2DNum_K = 40
train_layouts = 2  # 模拟拓扑结构变化
train_timesteps = 3
config = init_parameters()

print('Data generation')
# Data generation
# Train data
layouts, train_dists = D2D.layouts_generator(config, train_layouts)  # 创建训练集个数的tx、rx分布以及所有链路的距离信息（相当于生成训练集个数的地图）
# train_path_losses = D2D.compute_path_losses(config,train_dists)  # 计算所有链路的路径损耗的绝对值，这里的loss是path_loss，不是loss_function
train_path_losses = D2D.compute_path_losses_easily(config,train_dists)  # 使用WCNC的公式简易计算
train_channel_losses = D2D.add_fast_fading_sequence(train_timesteps, train_path_losses)  # 在每一个帧加入快衰落,将frame_num设为1


# 简易版信道生成
# 来自GNN4Com中的D2D代码
c = 1/np.sqrt(2)
train_channel_losses_2 = np.abs(c * np.random.randn(train_layouts, train_timesteps, D2DNum_K, D2DNum_K) + c * 1j * np.random.randn(train_layouts, train_timesteps, D2DNum_K, D2DNum_K))
print('end')