import numpy as np
import matplotlib.pyplot as plt
import math


def viz_1d(o, canvas=plt):
    canvas.imshow(o.numpy(), cmap="gray", aspect=o.shape[1] / 10)


def viz_3d(o, convas=plt):
    n_fea = o.shape[2]
    row_w = int(math.sqrt(n_fea))
    rows = []
    empt_mat = np.zeros((o.shape[0], o.shape[1]))
    for stitch_i in range(math.ceil(n_fea / row_w)):
        ret_mat = o[:, :, stitch_i * row_w]
        for stitch_j in range(1, row_w):
            idx = stitch_i * row_w + stitch_j
            if idx < n_fea:
                cur_mat = o[:, :, stitch_i * row_w + stitch_j]
            else:
                cur_mat = empt_mat
            ret_mat = np.concatenate((ret_mat, cur_mat), axis=1)
        rows.append(ret_mat)

    ret_mat = rows[0]
    for r in rows[1:]:
        ret_mat = np.concatenate((ret_mat, r), axis=0)
    convas.imshow(ret_mat)  # , cmap='gray')


def vizn_1d(o):
    n = len(o)
    fig, axs = plt.subplots(n, 1, figsize=(10, 5))
    for i in range(n):
        viz_1d(o[i : i + 1, :], axs[i])
        axs[i].axis("off")
    plt.show()


def vizn_3d(o):
    n = len(o)
    fig, axs = plt.subplots(1, n, figsize=(10, 5))
    for i in range(n):
        viz_3d(o[i], axs[i])
        axs[i].axis("off")
    plt.show()
