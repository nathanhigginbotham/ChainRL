import or_gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def create_vec_env(name, n_envs=4, env_seed=0, vec_cls=DummyVecEnv):
    '''
    maybe a bit clunky with so many wrapped fuctions
    '''
    def make_env():    
        def _init():
            env = or_gym.make(name, seed_int=0)
            return env
        return _init
    
    env = make_vec_env(make_env(), n_envs=n_envs, seed=env_seed, vec_env_cls=vec_cls)
    env = VecMonitor(env, './logs/')
    return env



class EvalModel:
    def __init__(self, env, model):
        self.env = env
        self.model = model


    def eval(self, num_evals, ep_len=30):

        obs = self.env.reset()
        raw_rewards = np.zeros((num_evals, 30))
        x = np.linspace(1,30,30)

        for i in range(num_evals):
            for j in range(ep_len):
                action = self.model.predict(obs)
                obs, reward, _, _ = self.env.step(action[0])
                raw_rewards[i,j] = reward 
            obs = self.env.reset()

        return raw_rewards


    def reward_stats(self, raw_rewards, filepath=None):

        mean_rewards = np.mean(raw_rewards, axis=0)
        std_rewards = np.std(raw_rewards, axis=0)
        upper_rewards = mean_rewards + std_rewards
        lower_rewards = mean_rewards - std_rewards


        upper_cumrewards = []
        lower_cumrewards = []
        cumrewards = []

        for i, v in enumerate(mean_rewards):
            cumreward = np.sum(mean_rewards[:i])
            cumrewards.append(cumreward)
            upper_cumrewards.append(cumreward+np.sqrt(np.sum(std_rewards[:i])))
            lower_cumrewards.append(cumreward-np.sqrt(np.sum(std_rewards[:i])))

        cumrewards = np.array(cumrewards) 
        upper_cumrewards = np.array(upper_cumrewards) 
        lower_cumrewards = np.array(lower_cumrewards) 

        if filepath != None:
            
            sac_eval_results = {'mean_rewards': mean_rewards, 'std_rewards': std_rewards, 
                'upper_rewards': upper_rewards, 'lower_rewards': lower_rewards,
                'cumlative_rewards': cumrewards, 'cumlative_lower_rewards': lower_cumrewards,
                'cumlative_upper_rewards': upper_cumrewards}       

            eval_df = pd.DataFrame(sac_eval_results)
            eval_df.to_csv(filepath, index_label='step')


        return mean_rewards, std_rewards, upper_rewards, lower_rewards, cumrewards, upper_cumrewards, lower_cumrewards

    def plot_rewards(self, raw_rewards, filepath=None):
        #probably add a save feature
        mean_rewards, _, upper_rewards, lower_rewards, cumrewards, upper_cumrewards, lower_cumrewards = self.reward_stats(raw_rewards)
        x = np.linspace(1, len(mean_rewards), len(mean_rewards))
        
        
        fig, ax1 = plt.subplots(figsize=(10,5))
        ax2 = ax1.twinx()
        
        ax1.fill_between(x, upper_cumrewards, lower_cumrewards, color='dodgerblue', alpha=0.4)
        lns1 = ax1.plot(x, cumrewards, color='dodgerblue', label='Cumlative Reward')
        
        ax2.fill_between(x, upper_rewards, lower_rewards, color='orange', alpha=0.4)
        lns2 = ax2.plot(x, mean_rewards, color='orange', label='Daily Reward')

        ax1.set_title('Cumlative and Daily Rewards')
        ax1.set_ylabel('Cumlative reward')
        ax2.set_ylabel('Daily day')
        ax1.set_xlabel('Day')
        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        
        
        # ax1.legend()
        # ax2.legend()

        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns,labs, loc=1)



        if filepath != None:
            plt.savefig(filepath)
        else:
            plt.show()