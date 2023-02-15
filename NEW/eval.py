import pandas as pd  
import tqdm 
import or_gym
import argparse
import os
import numpy as np

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env

from dictionary import algorithm_dictionary # This is a dictionary with the algorithms and their parameters
from dictionary import environment_dictionary # This is a dictionary with the environments and their parameters

from utils import GetFilePath # This is a function that creates a filepath based on the environment, model and hyperparameters passed as arguments

from utils import CreateEnv # This is a function that creates an environment based on an environment string passed as argument





def parse_args():
    """ Parses arguments passed to the script

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--algo', type=str, default='PPO', 
        help='Key of algorithm to be used')
    parser.add_argument('--env', type=str, default='IM1', 
        help='Key of environment to be used')
    
    args = parser.parse_args()
    return args



def GetBestModel(alg_string, filepath):
    """ Gets the best model from a given filepath

    Args:
        filepath (str): Path to the directory where the model is saved

    Returns:
        _type_: _description_
    """
    model = algorithm_dictionary[alg_string][0].load(filepath + 'best_model')
    return model


#################################
"""
Create and initialize all variables and containers.
 Nomenclature:
            I = On hand inventory at the start of each period at each stage (except last one).
            T = Pipeline inventory at the start of each period at each stage (except last one).
            R = Replenishment order placed at each period at each stage (except last one).
            D = Customer demand at each period (at the retailer)
            S = Sales performed at each period at each stage.
            B = Backlog at each period at each stage.
            LS = Lost sales at each period at each stage.
            P = Total profit at each stage.
            X = Initial inventory
            Y = 
        '''    
"""




def GetData(env, model, n_eval_episodes, filepath):
    """Gets data from the model interacting with the environment

    Args:
        env (_type_): _description_
        n_eval_episodes (_type_): _description_
    """
    obs = env.reset()
    
    # Here one may pull data listed above in the nomencalture section by adding the respective data name to the list below
       
    df_names = ['D', 'X', 'R', 'P', 'Y'] # Initialising DataFrame names
    # ['Market Demand', 'Inventory Node Stock', 'Reorder Amount', 'Node Profit', 'Edge Quantities'] Translated from above

    for episode in range(n_eval_episodes):
        for timestep in range(env.num_periods):
            action = model.predict(obs)
            obs, reward, _, _ = env.step(action[0])

        for df_name in df_names:
            df = getattr(env, df_name) # Get the DataFrame from the environment for the specific data
            
            if not os.path.exists(filepath + f'eval/{episode}/'): # If the directory does not exist, create it
                os.makedirs(filepath + f'eval/{episode}/')
            
            df.to_csv(filepath + f'eval/{episode}/{df_name}.csv') # Save the DataFrame as a csv file
            
        obs = env.reset() # Reset the environment for the next episode
        env.seed_int = episode # Select a new seed for the next episode
        
        
        
def GetMeanReward(env, model, n_eval_episodes, filepath):
    """Gets the mean reward from the model interacting with the environment

    It saves this into an evaluations.csv file in the filepath directory
    
    This also includes other parameters such as: mean_rewards, std_rewards, upper_rewards, 
                                                lower_rewards, cumrewards, lower_cumrewards, upper_cumrewards}


    Args:
        env (Gym Environment): Environment to be used
        model (Stable Baselines Model): Model to be used
        n_eval_episodes (_type_): Number of evaluation episodes
        filepath (str): Path to the directory where the model is saved (determined by the environment, model and hyperparameters keys)
    """
    
    obs = env.reset()
    env_periods = env.num_periods
    
    raw_rewards = np.zeros((n_eval_episodes, env_periods))
    x = np.linspace(1, env_periods, env_periods)

    for i in range(n_eval_episodes):
        for j in range(env_periods):
            action = model.predict(obs)
            obs, reward, _, _ = env.step(action[0])
            raw_rewards[i,j] = reward 
        obs = env.reset()

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

            
    eval_results = {'mean_rewards': mean_rewards, 'std_rewards': std_rewards, 
                'upper_rewards': upper_rewards, 'lower_rewards': lower_rewards,
                'cumlative_rewards': cumrewards, 'cumlative_lower_rewards': lower_cumrewards,
                'cumlative_upper_rewards': upper_cumrewards}       

    eval_df = pd.DataFrame(eval_results)
    
    savepath = filepath + 'eval/eval_results.csv'
    
    eval_df.to_csv(savepath, index_label='step')
    return



def Evaluate(env_string=None, alg_string=None, hyper_string=None, **kwargs):
    
    filepath = GetFilePath(env_string, alg_string, hyper_string)
    
    model = GetBestModel(alg_string, filepath)
    
    env = CreateEnv(env_string)
    
    # Change 10 to change how many episodes to evaluate over
    
    GetMeanReward(env, model, 10, filepath) # Get mean reward for model
    GetData(env, model, 10, filepath) # Get data from the model interacting with the environment
    
    return




if __name__ == '__main__':
    #### Code here to be executed
    args = parse_args()
    
    Evaluate(env_string=args.env, alg_string=args.algo, hyper_string='default')

    print('0')

