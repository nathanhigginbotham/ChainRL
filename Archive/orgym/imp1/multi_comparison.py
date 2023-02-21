import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
        
        
def eval_comparison(alg_1, alg_2, filepath=None):
    '''
    Probably should change into some kind of loop when
    we want more algorithms
    '''
    fig, ax1 = plt.subplots(figsize=(15,8))
    ax2 = ax1.twinx()
    
    
    
    x = np.linspace(1, len(ppo_eval['mean_rewards']), len(ppo_eval['mean_rewards']))

    ax1.fill_between(x, ppo_eval['cumlative_upper_rewards'], ppo_eval['cumlative_lower_rewards'], color='dodgerblue', alpha=0.4)
    lns1 = ax1.plot(x, ppo_eval['cumlative_rewards'], color='dodgerblue', label='PPO Cumlative Reward')
    ax2.fill_between(x, ppo_eval['upper_rewards'], ppo_eval['lower_rewards'], color='orange', alpha=0.4)
    lns2 = ax2.plot(x, ppo_eval['mean_rewards'], color='orange', label='PPO Daily Reward')

    ax1.fill_between(x, sac_eval['cumlative_upper_rewards'], sac_eval['cumlative_lower_rewards'], color='red', alpha=0.4)
    lns3 = ax1.plot(x, sac_eval['cumlative_rewards'], color='red', label='SAC Cumlative Reward')
    ax2.fill_between(x, sac_eval['upper_rewards'], sac_eval['lower_rewards'], color='limegreen', alpha=0.4)
    lns4 = ax2.plot(x, sac_eval['mean_rewards'], color='limegreen', label='SAC Daily Reward')

    ax1.set_title('Comparison of Cumlative and Daily Rewards')
    ax1.set_ylabel('Cumlative reward')
    ax2.set_ylabel('Daily day')
    ax1.set_xlabel('Day')
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    lns = lns1+lns2+lns3+lns4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns,labs, loc=9)
    
    # ax1.legend(loc=1)
    # ax2.legend(loc=0)

    if filepath != None:
        plt.savefig(filepath)
    else:
        plt.show()



def train_comparison(ppo_train, sac_train, filepath=None):
    '''
    train comparisons maybe needs to be turned into a loop too
    '''
    
    
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(ppo_train['episodes'][2::3], ppo_train['rewards'][2::3], '-', linewidth=0.5, color='dodgerblue')
    ax.plot(sac_train['episodes'][2::3], sac_train['rewards'][2::3], '-', linewidth=0.5, color='orange')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Reward of each episode as we train')
    ax.spines[['top', 'right']].set_visible(False)
    
    if filepath != None:
        plt.savefig(filepath)
    else:
        plt.show()


def train_time(ppo_train, sac_train,):
    '''
    think I have computed times wrong
    '''
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(ppo_train['episodes'], ppo_train['time'], label='PPO')
    ax.plot(sac_train['episodes'], sac_train['time'], label='SAC')
    plt.legend()
    plt.show()


def PlotAll(alg_strings, filepath):
    
    '''
    Function to plot the cumulative reward for all 7 algorithms. Or any number. Just input a list of strings.
    '''
    
    
    fig, axs = plt.subplots(4, 2, figsize=(15,8), squeeze=False, sharex=False, sharey=False)
    fig.delaxes(axs[3,1])
    
    x = np.linspace(1, len(eval_frames[0]['cumlative_rewards']), len(eval_frames[0]['cumlative_rewards']))
    
    colours = ['dodgerblue', 'salmon', 'forestgreen', 'crimson', 'orange', 'darkviolet', 'hotpink']
    
    first = axs[0,0]
    first.set_ylabel("Cumulative Reward (Profit)")
    first.yaxis.set_label_coords(-.1, -1.5)
    
    
    for i, alg in enumerate(alg_strings):
        if i<4:
            print(i)
            axes = axs[i,0]
            axes.plot(x, eval_frames[i]['cumlative_rewards'], color=colours[i], label=alg_strings[i])
            axes.fill_between(x, eval_frames[i]['cumlative_upper_rewards'], eval_frames[i]['cumlative_lower_rewards'], color=colours[i], alpha=0.4)
        else: 
            i = i-4
            axes = axs[i,1]
            i = i+4
            axes.plot(x, eval_frames[i]['cumlative_rewards'], color=colours[i], label=alg_strings[i])
            axes.fill_between(x, eval_frames[i]['cumlative_upper_rewards'], eval_frames[i]['cumlative_lower_rewards'], color=colours[i], alpha=0.4)
    
    
    
                
    fig.legend(bbox_to_anchor=(0.89,0.27), ncol=3, fontsize = 'x-large')
    
    filepath = os.getcwd() #### FIX THIS
    
    
    
    plt.xlabel("Day")
    
    plt.rcParams['savefig.dpi'] = 256
    plt.savefig(filepath +'/best_algs.pdf')
    
    plt.show()
    
    
    
    
    
    # ax2 = ax1.twinx()
    # x = np.linspace(1, len(ppo_eval['mean_rewards']), len(ppo_eval['mean_rewards']))

    # ax1.fill_between(x, ppo_eval['cumlative_upper_rewards'], ppo_eval['cumlative_lower_rewards'], color='dodgerblue', alpha=0.4)
    # lns1 = ax1.plot(x, ppo_eval['cumlative_rewards'], color='dodgerblue', label='PPO Cumlative Reward')
    # ax2.fill_between(x, ppo_eval['upper_rewards'], ppo_eval['lower_rewards'], color='orange', alpha=0.4)
    # lns2 = ax2.plot(x, ppo_eval['mean_rewards'], color='orange', label='PPO Daily Reward')

    # ax1.fill_between(x, sac_eval['cumlative_upper_rewards'], sac_eval['cumlative_lower_rewards'], color='red', alpha=0.4)
    # lns3 = ax1.plot(x, sac_eval['cumlative_rewards'], color='red', label='SAC Cumlative Reward')
    # ax2.fill_between(x, sac_eval['upper_rewards'], sac_eval['lower_rewards'], color='limegreen', alpha=0.4)
    # lns4 = ax2.plot(x, sac_eval['mean_rewards'], color='limegreen', label='SAC Daily Reward')

    # ax1.set_title('Comparison of Cumlative and Daily Rewards')
    # ax1.set_ylabel('Cumlative reward')
    # ax2.set_ylabel('Daily day')
    # ax1.set_xlabel('Day')
    # ax1.spines['top'].set_visible(False)
    # ax2.spines['top'].set_visible(False)
    
    # lns = lns1+lns2+lns3+lns4
    # labs = [l.get_label() for l in lns]
    # ax1.legend(lns,labs, loc=9)
    
    # # ax1.legend(loc=1)
    # # ax2.legend(loc=0)

    # if filepath != None:
    #     plt.savefig(filepath)
    # else:
    #     plt.show()
    
    
def TriplePlot(alg_strings, colours,filepath):
    
    '''
    Function to plot the cumulative reward for 3 algorithms. Or any number. Just input a list of strings.
    '''
    
    
    plt.figure(figsize=(15,8))
    
    x = np.linspace(1, len(eval_frames[0]['cumlative_rewards']), len(eval_frames[0]['cumlative_rewards']))
    
    plt.xlabel("Day")
    plt.ylabel("Cumulative Reward (Profit)")
    
    for i in range(len(alg_strings)):
        plt.plot(x, eval_frames[i]['cumlative_rewards'], color=colours[i], label=alg_strings[i])
        plt.fill_between(x, eval_frames[i]['cumlative_upper_rewards'], eval_frames[i]['cumlative_lower_rewards'], color=colours[i], alpha=0.4)
    
    
    plt.legend(ncol=3, fontsize = 'x-large')
                
    #fig.legend(bbox_to_anchor=(0.89,0.27), ncol=3, fontsize = 'x-large')
    
    filepath = os.getcwd() #### FIX THIS
    
    
    plt.xlabel("Day")
    
    plt.rcParams['savefig.dpi'] = 900
    plt.savefig(filepath + 'triplecomp.png')
    
    plt.show()
    
    
    
    

if __name__ == '__main__':

    
    
    alg_strings = ["SAC", "A2C", "PPO", "ARS", "RecurrentPPO", "TQC", "TRPO"] # Strings of All Algorithms -> Use PlotAll


    colours = ['dodgerblue', 'salmon', 'forestgreen', 'crimson', 'orange', 'darkviolet', 'hotpink'] # for consistency
    
    
    # alg_strings = ["ARS", "RecurrentPPO", "TRPO"]
    # colours = ['crimson', 'orange', 'hotpink'] # for consistency

    


    train_frames = []
    eval_frames = []
    
    
    
    logdir = '/net_nb_logs/'
    #main_path = os.getcwd()
    main_path = '/Users/nathan/Documents/GitHub/ChainRL'
    
    
    plt.rcParams['savefig.dpi'] = 256
    
    plt.style.use('seaborn-darkgrid')
    
    for i in alg_strings:
        train_frames.append(pd.read_csv(main_path + logdir + i + '_train_results.csv'))
        eval_frames.append(pd.read_csv(main_path + logdir + i + '_eval_results.csv'))
 
    
    
    
    #TriplePlot(alg_strings=alg_strings, colours=colours, filepath=logdir)
    PlotAll(alg_strings, logdir)
    
