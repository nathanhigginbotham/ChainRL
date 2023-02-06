import gym
import or_gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, SAC, A2C, DDPG, DQN, TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import torch as th


'''
Algorithms to examine!

On Policy:
    - PPO
    - A2C
    - TRPO

Off Policy:
    - SAC
    - DDPG
    - TQC
    - TD3


Features to look at:
    - Use of tensorboard for agent training
    - 

'''




name = 'InvManagement-v1'
num_envs = 4
total_timesteps = 50000
model_seed = 42


#here we define an augmentation to the network archecture which sits
#on top of the feature extractor, one for the actor and one for the critic
#SB3 will automatically add the correct last layer for the actor and critic there after
#here they share the layer 128 before splitting into their own layers which are not shared
#for off policy algorithms the vf is replaced with qf
policy_kwargs = dict(activation_fn=th.nn.ReLU, 
                    net_arch=[128, dict(pi=[32,32], vf=[16,8])])





#find difference between the two `DummyVecEnv` (usually faster) and `SubprocVecEnv   
# DummyvecEnv does not support multiprocesssing   
vec_env = make_vec_env(env_id=name, n_envs=num_envs, seed=0, vec_env_cls=DummyVecEnv)
single_env = or_gym.make(name)
eval_env = or_gym.make(name)
eval_callback = EvalCallback(eval_env=eval_env, best_model_save_path='./logs/',
                            log_path='.logs/', eval_freq=total_timesteps/10,
                            deterministic=True, render=False)

'''
Should we normalize the input features and rewards?
env = VecNormalize(env, norm_obs=True, norm_reward=True,
                   clip_obs=10.)

should we implement action noise?
which implement action noise:
    - TD3

HER replay buffer who it is appliacable to and 
how best to implement 

learning rate schedules, can also schedule PPO clip schedule
SB3 does not have schedules at this point but RL zoo does

look into use with isaac gym or envpool, envpool only linux based
no mac or windows distributions, isaac gym maybe more suitable
'''

#maybe look into saving the policy too revert back to intro notes
def save_model(model, filename, replay_buffer=False, replay_buffer_filename=None):
    if replay_buffer:
        assert replay_buffer_filename != None
        return model.save(filename), model.save_replay_buffer(replay_buffer_filename)
    else:
        return model.save(filename)

def load_model(model, model_filename, replay_filename=None):
    if replay_filename != None:
        return model.load(model_filename).load_replay_buffer(replay_filename)
    else:
        return model.load(model_filename)
        



def a2c_model_fn(env):
    '''
    Provide some explanation of algorithm ...
    
    variables of interest to use to potentially tune:
        - n_steps, gamma, gae_lambda, ent_coef, vf_coef,
        max_grad_norm, normalize_advantage, policy_kwargs

    also may want to explore types of learning rate schedulers
    '''
    model = A2C(policy='MlpPolicy', env=env, learning_rate=0.0005, n_steps=10,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.0001, vf_coef=0.5, max_grad_norm=0.8,
                rms_prop_eps=1e-5, use_rms_prop=True, use_sde=False, sde_sample_freq=-1, 
                normalize_advantage=True, policy_kwargs=policy_kwargs, verbose=0,
                seed=model_seed, device='cpu', _init_setup_model=True)

    model.learn(total_timesteps=50000, progress_bar=True, callback=eval_callback)
    return model


def ppo_model_fn(env):

    model = PPO('MlpPolicy', env, n_steps=30, n_epochs=10,
            batch_size=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            clip_range_vf=0.2, ent_coef=0.0001, max_grad_norm=0.5, 
            seed=0, verbose=0, policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=50000, progress_bar=True, callback=eval_callback)

    return model


def sac_model_fn(env):
    '''
    off policy algorithms will collect num_envs transitions per call to env.step
    gradient steps is number of gradient steps performed for each call to env.step
    setting this to -1 will result in num_envs gradient steps for every call to env.step   
    '''
    model = SAC('MlpPolicy', env, train_freq=1, gradient_steps=2, verbose=0)
    model.learn(total_timesteps=50000, progress_bar=True, callback=eval_callback)
    return model

def td3_model_fn(env):
    '''
    let us see  
    '''
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions))
    model = TD3('MlpPolicy', env,  action_noise=action_noise, verbose=0)
    model.learn(total_timesteps=50000, progress_bar=True, callback=eval_callback)
    return model


def dqn_model_fn(env):
    '''
    not working at present need to convert box to discrete to work not sure how yet
    '''
    model = DQN('MlpPolicy', env, verbose=0)
    model.learn(total_timesteps=50000, progress_bar=True, callback=eval_callback)
    return model



def ddpg_model_fn(env):
    #not working at present
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG('MlpPolicy', env, train_freq=num_envs, action_noise=action_noise, verbose=0)
    model.learn(total_timesteps=50000, progress_bar=True, callback=eval_callback)
    return model


ppo_model = ppo_model_fn(single_env)
mean_reward, std_reward = evaluate_policy(ppo_model, ppo_model.get_env(), n_eval_episodes=20)
print(f'mean reward from training: {mean_reward} +/- {std_reward}')




'''


td3_model = td3_model_fn(single_env)
mean_reward, std_reward = evaluate_policy(td3_model, td3_model.get_env(), n_eval_episodes=20)
print(f'mean reward from training: {mean_reward} +/- {std_reward}')


dqn_model = dqn_model_fn(env)
mean_reward, std_reward = evaluate_policy(dqn_model, dqn_model.get_env(), n_eval_episodes=20)
print(f'mean reward from training: {mean_reward} +/- {std_reward}')


ddpg_model = ddpg_model_fn(env)
mean_reward, std_reward = evaluate_policy(ddpg_model, ddpg_model.get_env(), n_eval_episodes=20)
print(f'mean reward from training: {mean_reward} +/- {std_reward}')


sac_model = sac_model_fn(env)
mean_reward, std_reward = evaluate_policy(sac_model, sac_model.get_env(), n_eval_episodes=20)
print(f'mean reward from training: {mean_reward} +/- {std_reward}')


ppo_model = ppo_model_fn(env)
mean_reward, std_reward = evaluate_policy(ppo_model, ppo_model.get_env(), n_eval_episodes=20)
print(f'mean reward from training: {mean_reward} +/- {std_reward}')

a2c_model = a2c_model_fn(env)
mean_reward, std_reward = evaluate_policy(a2c_model, a2c_model.get_env(), n_eval_episodes=20)
print(f'mean reward from training: {mean_reward} +/- {std_reward}')
'''
