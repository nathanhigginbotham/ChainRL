import numpy as np
import matplotlib.pyplot as plt
import pandas as pd





ppo_train = pd.read_csv('ppo/logs/ppo_train_results.csv')
sac_train = pd.read_csv('sac/logs/sac_train_results.csv')

plt.plot(ppo_train['episodes'], ppo_train['rewards'])
plt.plot(sac_train['episodes'], sac_train['rewards'])
plt.show()


ppo_eval = pd.read_csv('ppo/logs/ppo_eval_results.csv')
sac_eval = pd.read_csv('sac/logs/sac_eval_results.csv')



fig, ax = plt.subplots(figsize=(6,6))
ax.fill_between(ppo_eval['step'], ppo_eval['cumlative_upper_rewards'], ppo_eval['cumlative_lower_rewards'], alpha=0.7)
ax.plot(ppo_eval['step'], ppo_eval['cumlative_rewards'])

ax.fill_between(sac_eval['step'], sac_eval['cumlative_upper_rewards'], sac_eval['cumlative_lower_rewards'], alpha=0.7)
ax.plot(sac_eval['step'], sac_eval['cumlative_rewards'])

plt.show()
