import numpy as np
import matplotlib.pyplot as plt
import math


def viz_1d(o):
    plt.imshow(o.numpy(), cmap="gray", aspect=o.shape[1] / 10)
    plt.show()


def viz_3d(o):
    n_fea = o.shape[3]
    # row_w = 12
    row_w = int(math.sqrt(n_fea))
    rows = []
    empt_mat = np.zeros((o.shape[1], o.shape[2]))
    for stitch_i in range(math.ceil(n_fea / row_w)):
        ret_mat = o[0, :, :, stitch_i * row_w]
        for stitch_j in range(1, row_w):
            idx = stitch_i * row_w + stitch_j
            if idx < n_fea:
                cur_mat = o[0, :, :, stitch_i * row_w + stitch_j]
            else:
                cur_mat = empt_mat
            ret_mat = np.concatenate((ret_mat, cur_mat), axis=1)
        rows.append(ret_mat)

    ret_mat = rows[0]
    for r in rows[1:]:
        ret_mat = np.concatenate((ret_mat, r), axis=0)
    plt.imshow(ret_mat)  # , cmap='gray')
    plt.show()
