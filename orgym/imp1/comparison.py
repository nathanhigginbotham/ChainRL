import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
        
        
def eval_comparison(ppo_eval, sac_eval, filepath=None):
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


if __name__ == '__main__':


    ppo_train = pd.read_csv('ppo/logs/ppo_train_results.csv')
    ppo_eval = pd.read_csv('ppo/logs/ppo_eval_results.csv')
    sac_train = pd.read_csv('sac/logs/sac_train_results.csv')
    sac_eval = pd.read_csv('sac/logs/sac_eval_results.csv')

    eval_comparison(ppo_eval, sac_eval, 'comparison_figures/eval_comparison.png') 
    train_comparison(ppo_train, sac_train, 'comparison_figures/train_comparison.png')
    train_time(ppo_train, sac_train)