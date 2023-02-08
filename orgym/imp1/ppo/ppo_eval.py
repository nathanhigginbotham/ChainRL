import sys
sys.path.append('../')
print(sys.path)
from utils.study_model import EvalModel
import or_gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy





if __name__ == '__main__':

    env = or_gym.make('InvManagement-v1')

    final_model = PPO.load('./logs/final_model', env=env)
    mean_reward, std_reward = evaluate_policy(final_model, final_model.get_env(), n_eval_episodes=10)
    print(f'Mean cumlative reward on final model: {mean_reward}+/-{std_reward}')


    best_model = PPO.load('./logs/best_model', env=env)
    mean_reward, std_reward = evaluate_policy(best_model, best_model.get_env(), n_eval_episodes=10)
    print(f'Mean cumlative reward on best model: {mean_reward}+/-{std_reward}')


    model_study = EvalModel(env, best_model)
    rewards  = model_study.eval(20)
    mean_rewards, std_rewards, upper_rewards, lower_rewards, cumrewards, upper_cumrewards, lower_cumrewards = model_study.reward_stats(rewards)

    model_study.plot_rewards(rewards)


    data = pd.read_csv('logs/monitor.csv', names=['r','l','t'])
    data = data.drop(data.index[0:2])
    rewards = np.array(data['r']).astype(float)
    time = np.cumsum(np.array(data['t']).astype(float))
    length = int(data['l'].iloc[0])
    n_rows = len(data.index)
    episodes = np.linspace(start=1, stop=n_rows*length, num=n_rows)

    plt.plot(episodes, rewards, 'o')
    plt.show()

    plt.plot(episodes, time)
    plt.show()

    
    ppo_eval_results = {'mean_rewards': mean_rewards, 'std_rewards': std_rewards, 
                'upper_rewards': upper_rewards, 'lower_rewards': lower_rewards,
                'cumlative_rewards': cumrewards, 'cumlative_lower_rewards': lower_cumrewards,
                'cumlative_upper_rewards': upper_cumrewards}

    ppo_train_results = {'episodes': episodes, 'rewards': rewards, 'time': time}

    eval_df = pd.DataFrame(ppo_eval_results)
    train_df = pd.DataFrame(ppo_train_results)

    eval_df.to_csv('logs/ppo_eval_results.csv', index_label='step')
    train_df.to_csv('logs/ppo_train_results.csv', index=False)
    