import pandas as pd  
import tqdm 
import or_gym
import argparse


from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env

from dictionary import algorithm_dictionary # This is a dictionary with the algorithms and their parameters
from dictionary import environment_dictionary # This is a dictionary with the environments and their parameters


def get_filepath(env_string=None, model_string=None, hyper_string='default', **kwargs):
    """ Creates a filepath based on the environment, model and hyperparameters passed as arguments

    Args:
        env_string (str): Environment to be used. Defaults to None.
        model_string (str): Model to be used. Defaults to None.
        hyper_string (str): Hyperparameters to be used. Defaults to 'default'.

    Returns:
        _type_: _description_
    """
    
    if hyper_string == 'default':
        filepath = './Environments/' + env_string + '/' + model_string + '/default/'
    
    if hyper_string == 'tuned':
        filepath = './Environments/' + env_string + '/' + model_string + '/tuned/'
    
    return filepath


def create_env(env_string=None, env_seed=42, **kwargs): 
    """ Creates an environment based on an environment string passed as argument

    Args:
        env_string (str): Environment to be used. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    env = environment_dictionary[env_string][0]
    
    if environment_dictionary[env_string][1] != None:
        no_periods = environment_dictionary[env_string][1]
        
        env = or_gym.make(env, seed_int=env_seed, num_episodes=no_periods)
        
        return env
    
    env = or_gym.make(env, seed_int=env_seed)
    
    return env
