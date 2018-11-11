import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.ticker as ticker


def plot(data_name='score.pkl'):
    file_path = ['seq2seq', 'seq2seq-all']
    name_maps = ['Leap', 'KG-MIML-Net']

    df = pd.DataFrame()
    for idx, i in enumerate(file_path):
        df[name_maps[idx]] = pickle.load(open(os.path.join('saved', i, data_name), 'rb'))[0:30]
    ax = df.plot(title='Jaccard_similarity_score', ylim=[0, 0.4], style='-o')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.show()
    ax.get_figure().savefig('score.png')
    plt.close()


def plot_self_cmp():
    file_path = ['LR', 'Leap', 'seq2seq', 'Retain', 'DMNC', 'GAMENet_ja']
    name_maps = ['LR', 'Leap', 'Seq2Seq', 'RETAIN', 'DMNC', 'Ours']

    df = pd.DataFrame()
    for idx, i in enumerate(file_path):
        df[name_maps[idx]] = np.array(pickle.load(open(os.path.join('saved', i, 'history.pkl'), 'rb'))['ddi_rate'][0:20]) * 100
    ax = df.plot(ylim=[0, 25], style='-o')
    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('DDI RATE %', fontsize=13)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()
    ax.get_figure().savefig('exp_ddi_rate.eps', format='eps', dpi=1000)
    plt.close()


if __name__ == '__main__':
    plot_self_cmp()
