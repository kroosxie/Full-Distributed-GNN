import numpy as np


def get_directlink_losses(channel_losses):
    directlink_losses = []
    n = channel_losses.shape[0]
    m = channel_losses.shape[1]
    for i in range(n):
        directlink_losses.append(np.diagonal(channel_losses[i,:,:]))
    directlink_losses = np.array(directlink_losses)
    assert np.shape(directlink_losses)==(n, m)
    return directlink_losses
