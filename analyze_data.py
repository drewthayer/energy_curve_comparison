import os
import pandas as pd
import numpy as np
import datetime
import random
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from collections import Counter
import matplotlib.pyplot as plt

from DataTools.pickle import save_to_pickle, load_from_pickle

def n_random_integers(n, low=0, high=10):
    ''' generate random numbers with random.randint'''
    ii = []
    for i in range(n):
        ii.append(random.randint(low, high))
    return np.array(ii)

if __name__=='__main__':
    data = load_from_pickle('data','daily_array_all.pkl')
    # shape: (1440, 48)

    # cluster as gaussian mixture
    X = data
    n = 10
    gmm = GaussianMixture(n_components=n)
    gmm.fit(X)
    y = gmm.predict(X)
    probs = gmm.predict_proba(X)

    # sort results into clusters based on labels
    def clusters_from_lables(X,y):
        labels = np.unique(y)
        clusters = []
        for label in labels:
            clusters.append(X[np.where(y == label)])
        return clusters

    clusters = clusters_from_lables(X,y)
    cluster_sizes = [x.shape[0] for x in clusters]

    def calc_mean_clusters(clusters):
        '''clusters (list)      list of np.arrays '''
        mean_clusters = []
        for cluster in clusters:
            mean_clusters.append(np.mean(cluster, axis=0))
        return mean_clusters

    mean_clusters = calc_mean_clusters(clusters)

    def tightest_rectangle(n, tall=False):
        ''' inputs
            n       integer
            tall    flag, height > width '''
        a = np.ceil(np.sqrt(n))
        b = np.ceil(n/a)
        ax0 = int(min(a,b))
        ax1 = int(max(a,b))
        if tall:
            return ax1, ax0
        else:
            return ax0, ax1

    # plot cluster means
    def plot_mean_clusters(mean_clusters, cluster_sizes, title):
        title = 'mean cluster traces'
        m,n = tightest_rectangle(len(mean_clusters))
        y_max = max([max(x) for x in mean_clusters])

        fig, axs = plt.subplots(m,n)
        for ax, cluster, size in zip(axs.flatten(), mean_clusters, cluster_sizes):
            ax.plot(cluster)
            ax.set_ylim([0,y_max])
            ax.text(1,y_max*.8,'n={}'.format(size))

        plt.suptitle(title)
        plt.subplots_adjust(top=0.9)
        plt.show()

    plot_mean_clusters(mean_clusters, cluster_sizes, 'mean cluster traces')



    # assess GMM results
    # number of rows per cluster
    c = Counter(y)
    print(c)
    # plot
    xx = range(1,n+1)
    mean_proba = np.mean(probs, axis=0)
    sum_proba = np.sum(probs, axis=0)
    plt.bar(xx, mean_proba)
    plt.title('mean probabilities from gaussian mixture model, n={}'.format(n))
    plt.savefig('figs/gmm_mean_proba_n{}.png'.format(n), dpi=250)
    plt.close()
