import numpy as np


def layout_generator(general_para):
    N = general_para.n_links  # N是link的数目
    # first, generate transmitters' coordinates 首先，生成发射端的坐标
    tx_xs = np.random.uniform(low=0, high=general_para.field_length, size=[N, 1])  # 发射端的x坐标
    tx_ys = np.random.uniform(low=0, high=general_para.field_length, size=[N, 1])  # 发射端的y坐标
    while (True):  # loop until a valid layout generated
        # generate rx one by one rather than N together to ensure checking validity one by one
        # 逐个生成rx，而不是一起生成N个，确保可以逐个检查有效性
        rx_xs = [];
        rx_ys = []
        for i in range(N):
            got_valid_rx = False
            while (not got_valid_rx):
                pair_dist = np.random.uniform(low=general_para.shortest_directLink_length,
                                              high=general_para.longest_directLink_length)  # 生成D2D的direct_link的距离
                pair_angles = np.random.uniform(low=0, high=np.pi * 2)  # 生成direct_link的方位角
                rx_x = tx_xs[i] + pair_dist * np.cos(pair_angles)  # 生成rx的x坐标，cos用于向x轴投影
                rx_y = tx_ys[i] + pair_dist * np.sin(pair_angles)  # 生成rx的y坐标，cos用于向y轴投影
                if (0 <= rx_x <= general_para.field_length and 0 <= rx_y <= general_para.field_length):
                    got_valid_rx = True
            rx_xs.append(rx_x);
            rx_ys.append(rx_y)
        # For now, assuming equal weights and equal power, so not generating them 假设等功率等权重发射
        layout = np.concatenate((tx_xs, tx_ys, rx_xs, rx_ys), axis=1)  # layout（布局）：将坐标按列合并，即横向合并。
        distances = np.zeros([N, N])
        # compute distance between every possible Tx/Rx pair 计算每一个可能的收发对（包含干扰link）之间的距离
        for rx_index in range(N):
            for tx_index in range(N):
                tx_coor = layout[tx_index][0:2]  # [0:2]就是0，1，即tx的x、y坐标
                rx_coor = layout[rx_index][2:4]  # [2：4]就是2，3，即rx的x、y坐标
                # according to paper notation convention, Hij is from jth transmitter to ith receiver
                distances[rx_index][tx_index] = np.linalg.norm(
                    tx_coor - rx_coor)  # 求矩阵或向量的范数，linalg=linear+algebra，即线性代数
        # Check whether a tx-rx link (potentially cross-link) is too close 检查是否有相距过近的干扰对
        if (np.min(distances) > general_para.shortest_crossLink_length):
            break
    return layout, distances  # 返回tx、rx的位置坐标以及所有链路的距离


def layouts_generator(general_para, number_of_layouts):
    N = general_para.n_links  # D2D直连链路数目，即N个tx及N个rx
    print("<<<<<<<<<<<<<{} layouts: {}>>>>>>>>>>>>".format(
        number_of_layouts, general_para.setting_str))
    layouts = []
    dists = []
    for i in range(number_of_layouts):  # layouts：产生训练集个数的layout
        layout, dist = layout_generator(general_para)  # 返回tx、rx的位置坐标以及所有链路的距离
        layouts.append(layout)  # 将tx、rx的位置坐标堆叠到layouts中
        dists.append(dist)  # 将所有链路的距离堆叠到dists中
    layouts = np.array(layouts)  # 创建数组对象，便于后面reshape等操作
    dists = np.array(dists)
    # 断点语句：assert 是一种用来检测调试代码问题的语句，当条件为 True 时会直接通过，当为 False 时会抛出错误，可以用来定位和修改代码。
    assert np.shape(layouts)==(number_of_layouts, N, 4)
    assert np.shape(dists)==(number_of_layouts, N, N)
    return layouts, dists  # 返回训练集个数的tx、rx分布以及所有链路的距离信息（相当于生成训练集个数的地图）

def compute_path_losses_easily(general_para, distances):
    N = np.shape(distances)[-1]  # 直连链路的数目
    assert N == general_para.n_links  # 断点语句：结果为true直接通过
    pathlosses_dB = 38.46 + 20 * np.log10(distances)
    pathlosses = np.power(10, -(pathlosses_dB / 10))
    return pathlosses

def compute_path_losses(general_para, distances):
    N = np.shape(distances)[-1]  # 直连链路的数目
    assert N==general_para.n_links  # 断点语句：结果为true直接通过
    h1 = general_para.tx_height  # tx高度
    h2 = general_para.rx_height  # rx高度
    signal_lambda = 2.998e8 / general_para.carrier_f  # wavelength
    antenna_gain_decibel = general_para.antenna_gain_decibel  # 直连链路天线增益 decibel：dB
    # compute relevant quantity
    Rbp = 4 * h1 * h2 / signal_lambda  # threshold for a two-way path-loss model, LOS wave and reflected wave bp is breakpoint
    # LOS波和反射波双向路径损耗模型的阈值，bp是断点距离
    Lbp = abs(20 * np.log10(np.power(signal_lambda, 2) / (8 * np.pi * h1 * h2)))  # 基本断点距离
    # compute coefficient matrix for each Tx/Rx pair 计算每一个收发对的信道系数矩阵
    sum_term = 20 * np.log10(distances / Rbp)
    Tx_over_Rx = Lbp + 6 + sum_term + ((distances > Rbp).astype(int)) * sum_term  # adjust for longer path loss 注：dB形式
    pathlosses = -Tx_over_Rx + np.eye(N) * antenna_gain_decibel  # only add antenna gain for direct channel 仅在直连链路中加入天线增益 np.eye是对角元素
    pathlosses = np.power(10, (pathlosses / 10))  # convert from decibel to absolute 将dB形式转化为绝对值
    return pathlosses  # 返回所有链路的路径损耗的绝对值

def add_fast_fading_sequence(timesteps, train_path_losses):  # timesteps为帧的数目
    n = np.shape(train_path_losses)
    n_links = np.multiply(n[1],n[2])
    channel_losses_sequence = np.zeros((n[0],timesteps,n[1],n[2]))
    for i in range(n[0]):
        r = np.random.rand()
        alpha = np.resize(train_path_losses[i,:,:],n_links)
        noise_var = np.multiply(alpha,1-np.power(r,2))
        # channel coefficient matrix
        sims_real = np.zeros((timesteps,n_links))
        sims_imag = np.zeros((timesteps,n_links))
    # generate the channel coefficients for consecutive frames
    # 生成连续帧的信道系数
        sims_real[0,:] = np.random.normal(loc = 0, scale = np.sqrt(alpha))
        sims_imag[0,:] = np.random.normal(loc = 0, scale = np.sqrt(alpha))
        for j in range(timesteps-1):
            sims_real[j+1,:] = np.multiply(r,sims_real[j,:]) + np.random.normal(loc = 0, scale = np.sqrt(noise_var))
            sims_imag[j+1,:] = np.multiply(r,sims_imag[j,:]) + np.random.normal(loc = 0, scale = np.sqrt(noise_var))
        layout_channel_losses_sequence = (np.power(sims_real, 2) + np.power(sims_imag, 2))/2
        channel_losses_sequence[i,:,:,:] = np.resize(layout_channel_losses_sequence,(timesteps,n[1],n[2]))
    return channel_losses_sequence

# 真实版信道生成
# 来自GNN_over_the_air中的代码
def train_channel_generator_1(config, layouts, timesteps):
    layouts, train_dists = layouts_generator(config, layouts)  # 创建训练集个数的tx、rx分布以及所有链路的距离信息（相当于生成训练集个数的地图）
    # train_path_losses = D2D.compute_path_losses(config,train_dists)  # 计算所有链路的路径损耗的绝对值，这里的loss是path_loss，不是loss_function
    train_path_losses = compute_path_losses_easily(config, train_dists)  # 使用WCNC的公式简易计算
    train_channel_losses_1 = add_fast_fading_sequence(timesteps, train_path_losses)
    return train_channel_losses_1

# 简易版信道生成
# 来自GNN4Com中的D2D代码
def train_channel_generator_2(layouts, timesteps, D2DNum):
    c = 1 / np.sqrt(2)
    train_channel_losses_2 = np.abs(
        c * np.random.randn(layouts, timesteps, D2DNum, D2DNum) + c * 1j * np.random.randn(
            layouts, timesteps, D2DNum, D2DNum))
    return train_channel_losses_2