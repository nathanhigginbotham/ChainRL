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

# from tuned_params import tuned_params # This is a dictionary with the tuned parameters for each algorithm


"""
The idea of this script is to provide a general framework to train models in OpenAI Gym environments, this includes OR-Gym environments.

How do I use it?

    - It may be ran from both the command line and a jupyter notebook. Often it makes sense for it to be ran via the command line
    
    Via Command Line:
        - Still a work in progress to get flags set up, but for now you can just change the parameters in the main function.
    
    Via Jupyter Notebook:
        - Its probably best to import train.py as a module, and then call the functions from there.
        - RunScript would be the one to use to train inside a notebook..
    
    
How it works: 

    - It will generate an environment
    - Then it will generate a model
    - Then the callback will be generated
    - Then the model will be trained and saved


    
What does it do?

    - It will train a model in a given environment using a given algorithm and hyperparameters.
    - Once the model is trained, it will save it, and a best model, into the directory of the environment.
    
    EXAMPLE:
    
    If we were training a default PPO model in the InvManagement-v1 environment, the model would be saved in
    -> Environments -> InvManagement-v0 -> PPO -> -> default -> model.zip

    If we were training a tuned A2C model in the NetworkManagement-v0 environment, the model would be saved in
    -> Environment -> NetworkManagement-v0 -> A2C -> tuned -> model.zip


"""

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





def CreateEnv(env_string=None, env_seed=42, **kwargs): 
    """ Creates an environment based on an environment string passed as argument

    Args:
        env_string (str): Environment to be used. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    env = environment_dictionary[env_string][0]
    
    env = or_gym.make(env, seed_int=env_seed)
    
    
    return env


def CreateModel(alg_string=None, env=None, **kwargs): 
    """ Creates a model based on an algorithm, environment and hyperparameters passed as arguments

    Args:
        alg_string (str): Algorithm to be used. Defaults to None.
        env (Gym Environment): Environment to be used. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    
    alg, pol = algorithm_dictionary[alg_string][:2]
    
    model = alg(pol, env, **kwargs)
    
    return model


def Callback(env=None, env_string=None, model_string=None, hyper_string='default', eval_freq=10e3, **kwargs):
    """_summary_

    Args:
        env (Gym Environment):  Environment to be used. Defaults to None.
        env_string (str):  Name of Environment to be used. Defaults to None.
        model_string (str):  Name of Model to be used. Defaults to None.
        hyper_string (str): Name of Hyperparameters to be used. Defaults to 'default'.
        eval_freq (int, optional): Number of timesteps between evaluations. Defaults to 10,000.

    Returns:
        EvalCallback: Callback to be used during training
    """
    
    if hyper_string == 'default':
        filepath = './Environments/' + env_string + '/' + model_string + '/default'
    
    if hyper_string == 'tuned':
        filepath = './Environments/' + env_string + '/' + model_string + '/tuned'
    
    
    return EvalCallback(env, best_model_save_path=filepath, log_path=filepath, 
                                eval_freq=eval_freq, deterministic=True, render=False, verbose=1)


def TrainModel(model=None, env=None, total_timesteps=10e3, callback=None, **kwargs): 
    """ Trains a model based on an algorithm, environment and hyperparameters passed as arguments.

    Args:
        model (Stable Baselines Model): Model to be trained. Defaults to None.
        env (Gym Environment): Environment to be used. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    model.learn(total_timesteps, callback, progress_bar=True, **kwargs)
    
    return model

    

def RunScript(env_string=None, alg_string=None, hyper_string=None, filepath=None, **kwargs):
    """ Runs the script

    Args:
        env_string (str): Environment to be used. Defaults to None.
        alg_string (str): Algorithm to be used. Defaults to None.
        hyper_string (str): Hyperparameters to be used. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    # Create environment
    env = CreateEnv(env_string)
    
    
    
    env = Monitor(env, filename=filepath) 
    
    # Create model
    model = CreateModel(alg_string, env, **kwargs)
    
    # Create callback
    callback = Callback(env, env_string, alg_string, hyper_string, **kwargs)
    
    # Train model
    model = TrainModel(model, env, callback=callback, **kwargs)
    
    return model






if __name__ == '__main__':
    #### Code here to be executed
    
    args = parse_args()
    
    filepath = './Environments/' + args.env + '/' + args.algo + '/default'
    
    
    
    # env_string = environment_dictionary['NET1']
    # alg_string = algorithm_dictionary['ARS']
    
    # env_string = something
    # alg_string = something
    # hyper_string = something
    
    
    RunScript(args.env, args.algo, hyper_string='default', filepath=filepath)
    
    
    
    print('0')