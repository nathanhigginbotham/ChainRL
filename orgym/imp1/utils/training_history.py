import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class History:
    def __init__(self, filepath):

        self.filepath = filepath

        data = pd.read_csv('logs/monitor.csv', names=['r','l','t'])
        self.data = data.drop(data.index[0:2])
        self.rewards = np.array(self.data['r']).astype(float)
        self.time = np.cumsum(np.array(self.data['t']).astype(float))
        self.length = int(self.data['l'].iloc[0])
        self.n_rows = len(self.data.index)
        self.episodes = np.linspace(start=1, stop=self.n_rows*self.length, num=self.n_rows)


    def plot_history(self, filepath=None):

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(self.episodes[2::3], self.rewards[2::3], '-', linewidth=0.5, color='dodgerblue')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Reward of each episode as we train')
        ax.spines[['top', 'right']].set_visible(False)
        
        if filepath != None:
            plt.savefig(filepath)
        else:
            plt.show()

    def save(self, filepath):
        ppo_train_results = {'episodes': self.episodes, 'rewards': self.rewards, 'time': self.time}
        train_df = pd.DataFrame(ppo_train_results)
        train_df.to_csv(filepath, index=False)
