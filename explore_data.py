import os
import pandas as pd
import numpy as np
import datetime
import random
import matplotlib.pyplot as plt

from DataTools.pickle import save_to_pickle, load_from_pickle

def n_random_integers(n, low=0, high=10):
    ''' generate random numbers with random.randint'''
    ii = []
    for i in range(n):
        ii.append(random.randint(low, high))
    return np.array(ii)

if __name__=='__main__':
    df_30 = load_from_pickle('data','data_30min.pkl')

    # single timeseries of global power
    kw = df_30.Global_active_power # power in kW

    # clip to whole days
    firstday = '2006-12-17 00:00:00'
    lastday = '2010-11-25 23:30:00'

    kw = kw[kw.index >= firstday]
    kw = kw[kw.index <= lastday]

    # array of single day timeseries
    delta_t = kw.index[1] - kw.index[0] # size of timestep
    n_ts = int(datetime.timedelta(days=1) / delta_t) # number of timesteps
    n_rows = int(len(kw) / n_ts) # number of rows
    arr = kw.values.reshape(n_rows, n_ts)

    # plot n random days
    n_traces = 12
    ii = n_random_integers(n_traces, low=0, high=n_rows)
    sel = arr[ii,:]

    # plot
    xx = np.linspace(0,23.5,48)
    fig, axs = plt.subplots(4,3)
    for i, ax in enumerate(axs.flatten()):
        ax.plot(xx, sel[i,:])
        ax.set_ylim([0,120])
        # xticks: only on bottom row
        if i in [9,10,11]:
            ax.set_xticks(np.linspace(4,24,6))
            ax.set_xlabel('hour')
        else:
            ax.get_xaxis().set_ticks([])
        # yticks: only on left column
        if i in [0,3,6,9]:
            ax.set_yticks(np.linspace(50,150,3))
            ax.set_ylabel('kW')
        else:
            ax.get_yaxis().set_ticks([])

    titlestring = '{} random daily residential power timeseries'.format(n_traces)
    fig.get_axes()[0].annotate(titlestring, (0.5, 0.95),
                            xycoords='figure fraction', ha='center',
                            fontsize=18)
    plt.subplots_adjust(top=0.9)
    #plt.tight_layout()
    plt.show()
