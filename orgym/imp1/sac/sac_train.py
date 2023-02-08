import or_gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import torch as th

'''
PARAMETERS
'''

env_seed = 0
env_name = 'InvManagement-v1'
eval_env = or_gym.make(env_name, seed_int=42)
n_envs = 4
total_timesteps = 300000

sac_kwargs = {
    'learning_rate': 0.003,
    'buffer_size': 100000,
    'learning_starts':100,
    'batch_size': 256,
    'tau': 0.005,
    'train_freq': 1,
    'gradient_steps':1, #maybe set to -1
    'action_noise': None,
    'replay_buffer_class': None,
    'optimize_memory_usage': False,
    'ent_coef': 'auto',
    'target_update_interval': 1,
    'target_entropy': 'auto',
    'seed': 0,
    'policy_kwargs': dict(
                net_arch=dict(pi=[32,32], qf=[16,8]),
                activation_fn=th.nn.ReLU),
}


eval_callback = EvalCallback(eval_env=eval_env, best_model_save_path='logs/',
                            log_path='logs/', eval_freq=total_timesteps/(5*n_envs),
                            deterministic=True, render=False)


def create_vec_env(name, n_envs=4, seed=0, vec_cls=DummyVecEnv):
    '''
    maybe a bit clunky with so many wrapped fuctions
    '''
    def make_env():    
        def _init():
            env = or_gym.make(name)
            return env
        return _init
    
    env = make_vec_env(make_env(), n_envs=n_envs, seed=seed, vec_env_cls=vec_cls)
    env = VecMonitor(env, './logs/')
    return env


def sac_model_fn(env):

    model = SAC('MlpPolicy', env, **sac_kwargs)
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=eval_callback)

    return model





if __name__ == '__main__':
    #env = create_vec_env(name='InvManagement-v1', n_envs=n_envs, seed=env_seed)
    env = or_gym.make('InvManagement-v1', seed_int=0)
    model = sac_model_fn(env)
    model.save('logs/final_model')