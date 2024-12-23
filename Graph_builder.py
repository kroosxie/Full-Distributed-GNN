import numpy as np


def fully_connected_graph_builder(loss, norm_loss, K):
    x1 = np.expand_dims(norm_loss, axis=1)
    x2 = np.zeros((K, graph_embedding_size))
    x = np.concatenate((x1, x2), axis=1)
    x = torch.tensor(x, dtype=torch.float)

    # conisder fully connected graph
    loss2 = np.copy(loss)
    mask = np.eye(K)
    diag_loss2 = np.multiply(mask, loss2)
    loss2 = loss2 - diag_loss2
    attr_ind = np.nonzero(loss2)
    edge_attr = loss[attr_ind]
    edge_attr = np.expand_dims(edge_attr, axis=-1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    attr_ind = np.array(attr_ind)
    adj = np.zeros(attr_ind.shape)
    adj[0, :] = attr_ind[1, :]
    adj[1, :] = attr_ind[0, :]
    edge_index = torch.tensor(adj, dtype=torch.long)

    y = torch.tensor(np.expand_dims(loss, axis=0), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.contiguous(), edge_attr=edge_attr, y=y)
    return data

def build_graph():
    return


def proc_data(HH, norm_HH, K):
    n = HH.shape[0]  # layouts
    data_list = []
    for i in range(n):
        data = build_graph(HH[i, :, :], norm_HH[i, :], K)
        data_list.append(data)
    return data_list