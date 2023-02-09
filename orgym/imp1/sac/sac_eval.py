import sys
sys.path.append('../')
from utils.study_model import EvalModel
from utils.training_history import History
import gym
import or_gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.evaluation import evaluate_policy

if __name__ == '__main__':

    env = or_gym.make('InvManagement-v1')

    final_model = SAC.load('./logs/final_model', env=env)
    mean_reward, std_reward = evaluate_policy(final_model, final_model.get_env(), n_eval_episodes=10)
    print(mean_reward)
    print(std_reward)


    best_model = SAC.load('./logs/best_model', env=env)
    mean_reward, std_reward = evaluate_policy(best_model, best_model.get_env(), n_eval_episodes=10)
    print(mean_reward)
    print(std_reward)


    model_study = EvalModel(env, best_model)
    eval_rewards  = model_study.eval(20)
    mean_rewards, std_rewards, upper_rewards, lower_rewards, cumrewards, upper_cumrewards, lower_cumrewards = model_study.reward_stats(eval_rewards,
                                                                                                                'logs/sac_eval_results.csv')

    model_study.plot_rewards(eval_rewards, 'figures/eval_plot.png')


    train_history = History('logs/monitor.csv')
    train_history.plot_history('figures/train_plot.png')
    train_history.save('logs/sac_train_results.csv')