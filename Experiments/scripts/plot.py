import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import os

from utils import get_filepath
from utils import create_env

from scipy.stats import binned_statistic

plt.rcParams['figure.dpi'] = 256


def plot_network(env, ax):
    
    
    color = 'black'

    adjacency_matrix = np.vstack(env.graph.edges())
    # Set level colors
    level_col = {'retailer': 0,
                'distributor': 1,
                'manufacturer': 2,
                'raw_materials': 3}

    max_density = np.max([len(v) for v in env.levels.values()])
    node_coords = {}
    node_num = 1

    for i, (level, nodes) in enumerate(env.levels.items()):
        n = len(nodes)
        node_y = max_density / 2 if n == 1 else np.linspace(0, max_density, n)
        node_y = np.atleast_1d(node_y)
        ax.scatter(np.repeat(i, n), node_y, label=level, s=400, ec=color, fc='white', marker='o')

        for y in node_y:
            ax.annotate(r'$N_{}$'.format(node_num), xy=(i, y), ha='center', va='center', fontsize=10)
            node_coords[node_num] = (i, y)
            node_num += 1

    # Draw edges
    for node_num, v in node_coords.items():
        x, y = v
        sinks = adjacency_matrix[np.where(adjacency_matrix[:, 0]==node_num)][:, 1]
        for s in sinks:
            try:
                sink_coord = node_coords[s]
            except KeyError:
                continue
            
            x_ = np.hstack([x, sink_coord[0]])
            y_ = np.hstack([y, sink_coord[1]])
            ax.plot(x_, y_, color=color, linestyle='--', alpha=0.6, zorder=-10)

    ax.set_ylabel('Node')
    ax.set_yticks([0], [''])
    ax.set_xlabel('Level')
    ax.set_xticks(np.arange(len(env.levels)), [k for k in env.levels.keys()])
    ax.set_ymargin(0.2)


def visualise_network(env_string, filepath=None):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(8, 4)

    env = create_env(env_string)
    plot_network(env, ax)
    
    if filepath != None:
        save_dir = filepath + 'eval/plots/'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig.savefig(save_dir + 'network.pdf')
    fig.show()


# def plot_episode(env_name, algo_name, name):
#     save_path = f'./data/{env_name}/{algo_name}/{name}/'
#     env = make_env(env_name)

#     fig, ax = plt.subplots(nrows=4, ncols=2)
#     fig.set_size_inches(8, 8)

#     plot_network_on_ax(env, ax[0,0])

#     df_names = ['D', 'X', 'R', 'P', 'Y']
#     df_headers = [[0,1], [0], [0,1], [0], [0,1]]
#     data = {}

#     for i, df_name in enumerate(df_names):
#         df = pd.read_csv(save_path + f'eval/3/{df_name}.csv', index_col=[0], header=df_headers[i])
#         data[df_name] = df

#     cmap = plt.get_cmap('tab20')


#     for i, retailer_market in enumerate(data['D'].columns):
#         retailer, market = retailer_market
#         ax[0,1].plot(data['D'][retailer_market], linewidth=1.0, label=retailer_market)

#     for i, node_num in enumerate(data['X'].columns):
#         ax[1,0].plot(data['X'][node_num], linewidth=1.0, label=node_num, color=cmap(i))

#     for i, supplier_requester in enumerate(data['R'].columns):
#         supplier, requester = supplier_requester
#         ax[1,1].plot(data['R'][supplier_requester], linewidth=1.0, label=supplier_requester, color=cmap(i))

#     for i, node_num in enumerate(data['P'].columns):
#         ax[2,0].plot(data['P'][node_num], linewidth=1.0, label=node_num, color=cmap(i))
#         ax[2,1].plot(np.cumsum(data['P'][node_num]), linewidth=1.0, label=node_num, color=cmap(i))

#     Rtot = data['R'].groupby('Requester', axis=1).sum()

#     for i, node_num in enumerate(Rtot.columns):
#         ax[3,0].plot(Rtot[node_num], linewidth=1.0, label=node_num, color=cmap(i))

#     for i, source_reciever in enumerate(data['Y'].columns):
#         source, reciever = source_reciever
#         ax[3,1].plot(data['Y'][source_reciever], linewidth=1.0, label=source_reciever, color=cmap(i))

#     ax[0,1].legend(fontsize=6)
#     ax[1,0].legend(fontsize=6)
#     ax[1,1].legend(fontsize=6)
    
#     ax[0,1].set_xlabel('Timestep')
#     ax[0,1].set_ylabel('Market Demand')

#     ax[1,0].set_xlabel('Timestep')
#     ax[1,0].set_ylabel('Node Inventory Stock')

#     ax[1,1].set_xlabel('Timestep')
#     ax[1,1].set_ylabel('Supplier - Requester')

#     ax[2,0].set_xlabel('Timestep')
#     ax[2,0].set_ylabel('Node Profit')

#     ax[2,1].set_xlabel('Timestep')
#     ax[2,1].set_ylabel('Node Cumulative Profit')

#     ax[3,0].set_xlabel('Timestep')
#     ax[3,0].set_ylabel('Node Total Requested Stock')

#     ax[3,1].set_xlabel('Timestep')
#     ax[3,1].set_ylabel('Source - Reciever')


#     fig.tight_layout(h_pad=0.0, w_pad=0.0)

#     save_dir = f'./figures/{env_name}/'

#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     fig.savefig(save_dir + 'episode.pdf')
#     fig.show()

# #plot_network('NetworkManagement-v1-100')
# plot_episode('NetworkManagement-v1-100', 'TRPO', 'default')





def plot_training_curves(env_string, model_strings, xlims=None, ylims=None, filepath=None):
    """Plots the training curves for default models

    Args:
        env_strings (_type_): Environment key
        model_strings (Tuples): List of algorithm keys and hyper keys [algorithm_name, hyper_key]
        xlims (_type_): _description_
        ylims (_type_): _description_
        filepath (_type_, optional): _description_. Defaults to None.
    """
    
    
    # Example of model_strings = [('PPO', 'default'), 
    #                            ('TRPO', 'default'), ('RecurrentPPO', 'default'), ('A2C', 'default'), 
    #                              ('TQC', 'default'), ('ARS', 'default')]

    
    
    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.set_size_inches(8, 5)

    linewidth = 1.0
    alpha = 0.2

    cmap = plt.get_cmap('tab10')

    for i, model_name in enumerate(model_strings):
        algorithm_name, name = model_name
        data = pd.read_csv(f'../../Results/{env_string}/{algorithm_name}/default/monitor.csv', skiprows=1)

        num_bins = 100
        ep_binned_mean_r = binned_statistic(np.log10(np.arange(len(data['r'])) + 1), data['r'], 'mean', num_bins)
        ep_binned_std_r = binned_statistic(np.log10(np.arange(len(data['r'])) + 1), data['r'], 'std', num_bins)
        
        ax[0].plot(ep_binned_mean_r.bin_edges[1:], ep_binned_mean_r.statistic, 
            linewidth=linewidth, label=algorithm_name + ' ' + name, color=cmap(i))
        ax[0].fill_between(ep_binned_mean_r.bin_edges[1:], ep_binned_mean_r.statistic + ep_binned_std_r.statistic,
                           ep_binned_mean_r.statistic - ep_binned_std_r.statistic, color=cmap(i), alpha=alpha)

        num_bins = 100
        t_binned_mean_r = binned_statistic(np.log10(data['t']), data['r'], 'mean', num_bins)
        t_binned_std_r = binned_statistic(np.log10(data['t']), data['r'], 'std', num_bins)

        ax[1].plot(t_binned_mean_r.bin_edges[1:], t_binned_mean_r.statistic, 
            linewidth=linewidth, label=algorithm_name + ' ' + name, color=cmap(i))
        ax[1].fill_between(t_binned_mean_r.bin_edges[1:], t_binned_mean_r.statistic + t_binned_std_r.statistic,
                           t_binned_mean_r.statistic - t_binned_std_r.statistic, color=cmap(i), alpha=alpha)

        # ax[1].plot(data['t'], data['r'], linewidth=linewidth, alpha=0.4, color=colors[i]))

    ax[0].legend(fontsize=6)

    ax[0].set_xlabel('Log Episode')
    ax[0].set_ylabel('Reward')

    ax[1].set_xlabel('Log Walltime')
    ax[1].set_ylabel('Reward')

    if xlims is not None:
        ax[0].set_xlim(*xlims)
        ax[1].set_xlim(*xlims)

    if ylims is not None:
        ax[0].set_ylim(*ylims)
        ax[1].set_ylim(*ylims)

    fig.tight_layout()

    save_dir = f'../figures/{env_string}/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #fig.savefig(save_dir + 'training.pdf')
    fig.show()









if __name__ == '__main__':
    
    
    env_string = 'NET1_365p'
    alg_string = 'PPO'
    hyper_string = 'default'
    
    filepath = get_filepath(env_string, alg_string, hyper_string)
    
    plot_training_curves(env_string, [(alg_string, hyper_string)]) #, (1, 5), (-2500, 2500))
    
    
    #print(os.path.exists('../Results/{env_string}/{algorithm_name}/default/monitor.csv'))
    
    plt.show()