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

from utils import get_filepath # Gets file path from environment, model and hyperparameters

from utils import create_env # Creates an environment based on an environment string passed as argument



# from tuned_params import tuned_params # This is a dictionary with the tuned parameters for each algorithm


"""
The idea of this script is to provide a general framework to train models in OpenAI Gym environments, this includes OR-Gym environments.

How do I use it?

    - It may be ran from both the command line and a jupyter notebook. Often it makes sense for it to be ran via the command line
    
    Via Command Line:
        - The script can be ran from the command line using the following command:
        
        `python train.py --algo PPO --env IM1 --time 5e5` [To run for 500,000 timesteps on InvManagement-v1 with PPO]

        `python train.py --algo [ALGORITHM KEY] --env [ENVIRONMENT KEY] --time [NUMBER OF TIMESTEPS]` 
        [To run for [NUMBER OF TIMESTEPS] timesteps on [ENVIRONMENT KEY] with [ALGORITHM KEY]]
    
    
    Via Jupyter Notebook:
        - Its probably best to import train.py as a module, and then call the functions from there.
        - RunScript would be the one to use to train inside a notebook..
    
    
How it works: 

    - It will generate an environment
    - Then it will generate a model
    - Then the callback will be generated
    - Then the model will be trained and saved to the respective directory given by the environment and algorithm keys.


    
What does it do?

    - It will train a model in a given environment using a given algorithm and hyperparameters.
    - Once the model is trained, it will save it, and a best model, into the directory of the environment.
    
    EXAMPLE:
    
    If we were training a default PPO model in the InvManagement-v1 environment, the model would be saved in
    -> Environments -> IM1 -> PPO -> -> default -> model.zip

    If we were training a tuned A2C model in the NetworkManagement-v0 environment, the model would be saved in
    -> Environment -> NET0 -> A2C -> tuned -> model.zip


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
    parser.add_argument('--time', type=int, default=5e5, 
        help='Number of timesteps to train for')
    
    args = parser.parse_args()
    return args



def create_model(alg_string=None, env=None, **kwargs): 
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

def get_callback(env=None, env_string=None, model_string=None, hyper_string='default',filepath=None, eval_freq=10e3, **kwargs):
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
    
    return EvalCallback(env, best_model_save_path=filepath, log_path=filepath, 
                                eval_freq=eval_freq, deterministic=True, render=False, verbose=1)


# def CreateMonitor(env=None, env_string=None, model_string=None, hyper_string='default', **kwargs):
#     """ Creates a monitor to be used during training

#     Args:
#         env (Gym Environment):  Environment to be used. Defaults to None.
#         env_string (str):  Name of Environment to be used. Defaults to None.
#         model_string (str):  Name of Model to be used. Defaults to None.
#         hyper_string (str): Name of Hyperparameters to be used. Defaults to 'default'.

#     Returns:
#         _type_: _description_
#     """
    
#     if hyper_string == 'default':
#         filepath = './Environments/' + env_string + '/' + model_string + '/default/'
    
#     if hyper_string == 'tuned':
#         filepath = './Environments/' + env_string + '/' + model_string + '/tuned/'
    
#     return Monitor(env, filepath, allow_early_resets=True)


def train_model(model=None, env=None, total_timesteps=10e3, callback=None, **kwargs): 
    """ Trains a model based on an algorithm, environment and hyperparameters passed as arguments.

    Args:
        model (Stable Baselines Model): Model to be trained. Defaults to None.
        env (Gym Environment): Environment to be used. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    model.learn(total_timesteps, callback, progress_bar=True, **kwargs)
    
    return model
 

def run_script(env_string=None, alg_string=None, hyper_string=None, total_timesteps=None, **kwargs):
    """ Runs the script

    Args:
        env_string (str): Environment to be used. Defaults to None.
        alg_string (str): Algorithm to be used. Defaults to None.
        hyper_string (str): Hyperparameters to be used. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    # Gets the correct filepath for the model's training results
    filepath = GetFilePath(env_string, alg_string, hyper_string)
    
    # Create environment
    env = create_env(env_string)
    
    # env = CreateMonitor(env, env_string, alg_string, hyper_string, **kwargs)
    env = Monitor(env, filename=filepath) 
    
    # Create model
    model = create_model(alg_string, env, **kwargs)
    
    # Create callback
    callback = get_callback(env, env_string, alg_string, hyper_string, filepath, **kwargs)
    
    # Train model
    model = train_model(model, env, total_timesteps, callback=callback, **kwargs)
    
    return model



if __name__ == '__main__':
    #### Code here to be executed
    
    args = parse_args()
    
    
    ##### This filepath needs to be changed to be more dynamic
    ##### It needs to be generated from if the model is tuned or not.
    ##### It is defined here purely for the purposes of the monitor.
    ##### The evalcallback is already set up to save the model in the correct directory.
    
   #filepath = './Environments/' + args.env + '/' + args.algo + '/default'
    
    # env_string = environment_dictionary['NET1']
    # alg_string = algorithm_dictionary['ARS']
    
    # env_string = something
    # alg_string = something
    # hyper_string = something
    
    
    run_script(args.env, args.algo, hyper_string='default', total_timesteps=args.time)
    
    
    print('0')