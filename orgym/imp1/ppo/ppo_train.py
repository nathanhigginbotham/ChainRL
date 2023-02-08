import or_gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
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
ppo_kwargs = {
    'n_steps' : 30, #is this multiplied by n_envs before update begins?
    'n_epochs' : 10,
    'batch_size' : 10*n_envs, #we can make this bigger if n_evs increases the effective number of steps
    'gamma' : 0.99,
    'gae_lambda' : 0.95,
    'clip_range' : 0.2,
    'clip_range_vf' : 0.2, #might not want to clip
    'ent_coef' : 0.0001,
    'max_grad_norm' : 0.5,
    'seed' : 0,
    'policy_kwargs': dict(
                net_arch=[128, dict(pi=[32,32], vf=[16,8])],
                activation_fn=th.nn.ReLU,
                ortho_init=True),
}


eval_callback = EvalCallback(eval_env=eval_env, best_model_save_path='./logs/',
                            log_path='./logs/', eval_freq=total_timesteps/(5*n_envs),
                            deterministic=True, render=False)


def create_vec_env(name, n_envs=4, seed=0, vec_cls=DummyVecEnv):
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


def ppo_model_fn(env):

    model = PPO('MlpPolicy', env, **ppo_kwargs)
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=eval_callback)

    return model





if __name__ == '__main__':
    #find how often monitor is written, monitor is written to every n_steps since that is when the agent updates
    env = create_vec_env(name='InvManagement-v1', n_envs=n_envs, seed=env_seed)
    model = ppo_model_fn(env)
    model.save('logs/final_model')