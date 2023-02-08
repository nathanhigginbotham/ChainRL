import os
import optuna
import numpy as np
from typing import Dict, Any
from rl_zoo3 import linear_schedule

from torch import nn
from stable_baselines3 import PPO
import gym
import or_gym
from stable_baselines3.common.callbacks import EvalCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
import pickle as pkl
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import joblib

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    '''
    Sampling PPO hyperparameters
    '''


    batch_size = trial.suggest_categorical('batch_size', [5, 10, 15]) #batch_size*n_evs
    #batch_size = trial.suggest_categorical('batch_size', [20, 40, 60]) #batch_size*n_evs
    n_steps =  30
    gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.99, 0.999])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1)
    lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'costant']) #think there is a constant one too
    net_arch = trial.suggest_categorical('net_arch', ['first', 'second'])
    ent_coef = trial.suggest_loguniform('ent_coef', 0.000001, 0.01)
    clip_range = trial.suggest_categorical('clip_range', [0.1, 0.2, 0.3])
    n_epochs = trial.suggest_categorical('n_epochs', [3,5,10,15])
    gae_lambda = trial.suggest_categorical('gae_lambda', [0.8, 0.9, 0.95, 0.99])
    max_grad_norm = trial.suggest_categorical('max_grad_norm', [0.2, 0.5, 0.8])
    vf_coef = trial.suggest_uniform('vf_coef', 0.3, 0.8)
    ortho_init = trial.suggest_categorical('ortho_init', [True, False])
    activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu'])

    if batch_size > 4*n_steps:
        batch_size = n_steps

    if lr_schedule == 'linear':
        learning_rate = linear_schedule(learning_rate)



    net_arch = {
        'first': dict(pi=[64, 64], vf=[64, 64]),
        'second': dict(pi=[256], vf=[8])
    }[net_arch]

    activation_fn = {'tanh':nn.Tanh, 'relu':nn.ReLU}[activation_fn]

    return {
        'n_steps': n_steps,
        'batch_size': batch_size,
        'gamma': gamma,
        'learning_rate': learning_rate,
        'ent_coef': ent_coef,
        'clip_range': clip_range,
        'n_epochs': n_epochs,
        'gae_lambda': gae_lambda,
        'max_grad_norm': max_grad_norm,
        'vf_coef': vf_coef,


        'policy_kwargs': dict(
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init),
    }



def create_vec_env(name, n_envs=4, vec_cls=DummyVecEnv):
    '''
    maybe a bit clunky with so many wrapped fuctions
    '''
    def make_env():    
        def _init():
            env = or_gym.make(name, seed_int=0)
            return env
        return _init
    
    env = make_vec_env(make_env(), n_envs=n_envs, seed=0, vec_env_cls=vec_cls)
    return env




def objective(trial: optuna.Trial) -> float:
    
    hyperparameters = sample_ppo_params(trial)
    #env = create_vec_env(name='InvManagement-v1', n_envs=4)
    env = or_gym.make('InvManagement-v1')
    eval_env = or_gym.make('InvManagement-v1')
    total_timesteps = 300000

    ppo_model = PPO('MlpPolicy', env=env, seed=42, verbose=0, **hyperparameters)

    eval_callback = EvalCallback(eval_env=eval_env, #change to vec_env might not break then?
                            eval_freq=total_timesteps/10,
                            deterministic=True, render=False, verbose=0)

    trial_path = f'{study_path}/trial_{str(trial.number)}'
    os.makedirs(trial_path, exist_ok=True)

    with open(f'{trial_path}/hyperparameters.txt', 'w') as f:
        f.write(str(hyperparameters))

    ppo_model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    reward = eval_callback.best_mean_reward

    return reward



if __name__ == '__main__':
    
    study_path = 'logs/tuning'


    sampler = TPESampler(n_startup_trials=10, multivariate=True)
    pruner = MedianPruner(n_min_trials=10, n_warmup_steps=10)

    study = optuna.create_study(
        study_name='ppo_study',
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
        direction='maximize',
    )

    study.optimize(objective, n_jobs=4, n_trials=128, show_progress_bar=True)

    joblib.dump(study, 'ppo_study.pkl')

    best_trial = study.best_trial

    print(f'Best trial: {best_trial.number}')
    print(f'value: {best_trial.value}')
    print('Params:')
    for key, value in best_trial.params.items():
        print(f'    {key}: {value}')


    study.trials_dataframe().to_csv(f'{study_path}/report.csv')

    with open(f'{study_path}/study.pkl', 'wb+') as f:
        pkl.dump(study, f)

    fig1 = plot_optimization_history(study)
    fig2 = plot_parallel_coordinate(study)
    fig3 = plot_param_importances(study)

    fig1.show()
    fig2.show()
    fig3.show()