import numpy as np
import matplotlib
from utils import now
from model import MAGAN
from loader import Loader
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm
plt.ion(); fig = plt.figure()


def get_data(n_batches=2, n_pts_per_cluster=5000):
    """Return the artificial data."""
    make = lambda x,y,s: np.concatenate([np.random.normal(x,s, (n_pts_per_cluster, 1)), np.random.normal(y,s, (n_pts_per_cluster, 1))], axis=1)
    # batch 1
    xb1 = np.concatenate([make(-1.3, 2.2, .1), make(.1, 1.8, .1), make(.8, 2, .1)], axis=0)
    labels1 = np.concatenate([0 * np.ones(n_pts_per_cluster), 1 * np.ones(n_pts_per_cluster), 2 * np.ones(n_pts_per_cluster)], axis=0)

    # batch 2
    xb2 = np.concatenate([make(-.9, -2, .1), make(0, -2.3, .1), make(1.5, -1.5, .1)], axis=0)
    labels2 = np.concatenate([0 * np.ones(n_pts_per_cluster), 1 * np.ones(n_pts_per_cluster), 2 * np.ones(n_pts_per_cluster)], axis=0)

    return xb1, xb2, labels1, labels2

# Load the data
xb1, xb2, labels1, labels2 = get_data()
print("Batch 1 shape: {} Batch 2 shape: {}".format(xb1.shape, xb2.shape))

# Prepare the loaders
loadb1 = Loader(xb1, labels=labels1, shuffle=True)
loadb2 = Loader(xb2, labels=labels2, shuffle=True)
batch_size = 100

# Build the tf graph
magan = MAGAN(dim_b1=xb1.shape[1], dim_b2=xb2.shape[1], no_gpu=True)

# Train
for i in range(1, 100000):
    if i % 100 == 0: print("Iter {} ({})".format(i, now()))
    xb1_, labels1_ = loadb1.next_batch(batch_size)
    xb2_, labels2_ = loadb2.next_batch(batch_size)

    magan.train(xb1_, xb2_)

    # Evaluate the loss and plot
    if i % 500 == 0:
        xb1_, labels1_ = loadb1.next_batch(10 * batch_size)
        xb2_, labels2_ = loadb2.next_batch(10 * batch_size)

        lstring = magan.get_loss(xb1_, xb2_)
        print("{} {}".format(magan.get_loss_names(), lstring))


        xb1 = magan.get_layer(xb1_, xb2_, 'xb1')
        xb2 = magan.get_layer(xb1_, xb2_, 'xb2')
        Gb1 = magan.get_layer(xb1_, xb2_, 'Gb1')
        Gb2 = magan.get_layer(xb1_, xb2_, 'Gb2')

        fig.clf()
        axes = fig.subplots(2, 2, sharex=True, sharey=True)
        axes[0, 0].set_title('Original')
        axes[0, 1].set_title('Generated')
        axes[0, 0].scatter(0, 0, s=45, c='b', label='Batch 1'); axes[0, 0].scatter(0,0, s=100, c='w'); axes[0, 0].legend(handletextpad=.1, borderpad=.5, loc='center left', bbox_to_anchor=[.02, .5]);
        axes[0, 1].scatter(0, 0, s=45, c='r', label='Batch 2'); axes[0, 1].scatter(0,0, s=100, c='w'); axes[0, 1].legend(handletextpad=.1, borderpad=.5, loc='center left', bbox_to_anchor=[.02, .5]);
        axes[1, 0].scatter(0, 0, s=45, c='r', label='Batch 2'); axes[1, 0].scatter(0,0, s=100, c='w'); axes[1, 0].legend(handletextpad=.1, borderpad=.5, loc='center left', bbox_to_anchor=[.02, .5]);
        axes[1, 1].scatter(0, 0, s=45, c='b', label='Batch 1'); axes[1, 1].scatter(0,0, s=100, c='w'); axes[1, 1].legend(handletextpad=.1, borderpad=.5, loc='center left', bbox_to_anchor=[.02, .5]);

        for lab, marker in zip([0, 1, 2], ['x', 'D', '.']):
            axes[0, 0].scatter(xb1[labels1_ == lab, 0], xb1[labels1_ == lab, 1], s=45, alpha=.5, cmap=matplotlib.cm.jet, c='b', marker=marker)
            axes[0, 1].scatter(Gb2[labels1_ == lab, 0], Gb2[labels1_ == lab, 1], s=45, alpha=.5, cmap=matplotlib.cm.jet, c='r', marker=marker)
        for lab, marker in zip([0, 1, 2], ['x', 'D', '.']):
            axes[1, 0].scatter(xb2[labels2_ == lab, 0], xb2[labels2_ == lab, 1], s=45, alpha=.5, cmap=matplotlib.cm.jet, c='r', marker=marker)
            axes[1, 1].scatter(Gb1[labels2_ == lab, 0], Gb1[labels2_ == lab, 1], s=45, alpha=.5, cmap=matplotlib.cm.jet, c='b', marker=marker)
        fig.canvas.draw()
        plt.pause(1)











