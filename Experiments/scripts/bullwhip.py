import or_gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy as PPOPolicy

from scipy.stats import poisson, norm
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3.common.monitor import Monitor
import tqdm
import os

# from plot import eval_comparison, plot_network_on_ax, plot_episode
from scipy.stats import rv_discrete
from stable_baselines3.common.callbacks import EvalCallback

from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from dictionary import algorithm_dictionary

import matplotlib.pyplot as plt
from bullwhip_env import InvManagementMasterEnv

"""
To Do:

- explore alternative rl algorithms
- also add slump
- probably will move all plotting to notebook file and this will train and eval

"""


# Parameters of experiment
num_periods = 100
total_timesteps = 5000000
n_steps = num_periods
batch_size = int(n_steps / 20)
n_eval_episodes = 50
user_D = np.array(
    [
        25.0,
        18.0,
        18.0,
        19.0,
        21.0,
        20.0,
        14.0,
        22.0,
        22.0,
        17.0,
        22.0,
        16.0,
        18.0,
        15.0,
        22.0,
        21.0,
        20.0,
        23.0,
        28.0,
        22.0,
        17.0,
        18.0,
        17.0,
        27.0,
        25.0,
        19.0,
        22.0,
        19.0,
        36.39183958,
        43.56894222,
        50.11621268,
        55.38698078,
        58.8119204,
        60.0,
        58.8119204,
        55.38698078,
        50.11621268,
        43.56894222,
        36.39183958,
        17.0,
        13.0,
        21.0,
        16.0,
        23.0,
        26.0,
        19.0,
        23.0,
        23.0,
        18.0,
        16.0,
        21.0,
        17.0,
        32.0,
        23.0,
        12.0,
        23.0,
        23.0,
        22.0,
        22.0,
        26.0,
        22.0,
        16.0,
        23.0,
        19.0,
        23.0,
        23.0,
        14.0,
        23.0,
        21.0,
        16.0,
        21.0,
        24.0,
        29.0,
        17.0,
        21,
        21,
        19,
        19,
        32,
        24,
        12,
        19,
        14,
        20,
        17,
        19,
        21,
        29,
        18,
        23,
        22,
        16,
        17,
        21,
        16,
        22,
        19,
        15,
        22,
        15,
    ]
)


param_dictionary = {
    "SAC": {},
    "A2C": {},
    "PPO": {"n_steps": num_periods, "batch_size": batch_size},
    "ARS": {},
    "RecurrentPPO": {},
    "TQC": {},
    "TRPO": {},
}

#######################################################################
"""Market Demand Probability Distribution with Bullwhip effect"""


def eval_comparison(ppo_eval, filepath=None):
    """
    Probably should change into some kind of loop when
    we want more algorithms
    """
    fig, ax1 = plt.subplots(figsize=(15, 8))
    ax2 = ax1.twinx()
    x = np.linspace(1, len(ppo_eval["mean_rewards"]), len(ppo_eval["mean_rewards"]))

    ax1.fill_between(
        x,
        ppo_eval["cumlative_upper_rewards"],
        ppo_eval["cumlative_lower_rewards"],
        color="dodgerblue",
        alpha=0.4,
    )
    lns1 = ax1.plot(
        x,
        ppo_eval["cumlative_rewards"],
        color="dodgerblue",
        label="PPO Cumlative Reward",
    )
    ax2.fill_between(
        x,
        ppo_eval["upper_rewards"],
        ppo_eval["lower_rewards"],
        color="orange",
        alpha=0.4,
    )
    lns2 = ax2.plot(
        x, ppo_eval["mean_rewards"], color="orange", label="PPO Daily Reward"
    )

    ax1.set_title("Comparison of Cumlative and Daily Rewards")
    ax1.set_ylabel("Cumlative reward")
    ax2.set_ylabel("Daily day")
    ax1.set_xlabel("Day")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=9)

    # ax1.legend(loc=1)
    # ax2.legend(loc=0)

    if filepath != None:
        plt.savefig(filepath)
    else:
        plt.show()


def bullwhip_env():
    """Train environment which has the bullwhip effect"""
    bullwhip_train_env = InvManagementMasterEnv(
        periods=num_periods,
        dist=0,
        dist_param={"epsilon": 0.03, "amplitude": 50, "sigma": 5},
    )

    return bullwhip_train_env


def reg_env():
    env = or_gym.make("InvManagement-v1", periods=num_periods)
    return env


"""Evaluation Environment with bullwhip effect"""
# why are the market demands different?
def bullwhip_eval_env():
    eval_env = or_gym.make(
        "InvManagement-v1", periods=num_periods, dist=5, user_D=user_D
    )
    return eval_env


env_dictionary = {"regular": reg_env(), "eratic": bullwhip_env()}

########################################################################
"""Here we define the models to be implemented in the experiment"""


class BullWhipExperiment:
    def __init__(self, model_key, env, env_key):
        self.model_key = model_key
        self.env_key = env_key
        self.train_env = Monitor(
            env,
            f"Experiments/Experiment_data/bullwhip_data/{model_key}/{env_key}/monitor.csv",
        )

    def train(self):

        print(
            f"Now training {self.model_key} in the {self.env_key} environment to deal with bullwhip effect"
        )

        eval_callback = EvalCallback(
            eval_env=self.train_env,
            best_model_save_path=f"Experiments/Experiment_data/bullwhip_data/{self.model_key}/{self.env_key}",
            log_path=f"Experiments/Experiment_data/bullwhip_data/{self.model_key}/{self.env_key}",
            eval_freq=total_timesteps / (20),
            deterministic=True,
            render=False,
        )

        algorithm_hyperparams = param_dictionary[self.model_key]
        algorithm, policy = algorithm_dictionary[self.model_key][:2]
        bullwhip_model = algorithm(policy, self.train_env, **algorithm_hyperparams)

        bullwhip_model.learn(
            total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True
        )

        data = pd.read_csv(
            f"Experiments/Experiment_data/bullwhip_data/{self.model_key}/{self.env_key}/monitor.csv",
            names=["r", "l", "t"],
        )
        self.data = data.drop(data.index[0:2])
        rewards = np.array(self.data["r"]).astype(float)
        time = np.cumsum(np.array(self.data["t"]).astype(float))
        length = int(self.data["l"].iloc[0])
        n_rows = len(self.data.index)
        episodes = np.linspace(start=1, stop=n_rows * length, num=n_rows)

        train_results = {"episodes": episodes, "rewards": rewards, "time": time}
        train_results = pd.DataFrame(train_results)

        return bullwhip_model, train_results

    def plot_train_history(self, train_results):
        fig, ax = plt.subplots(figsize=(10, 4))
        episodes = train_results["episodes"]
        rewards = train_results["rewards"]
        ax.plot(episodes[2::3], rewards[2::3], "-", linewidth=0.5, color="dodgerblue")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Reward of each episode as we train")
        ax.spines[["top", "right"]].set_visible(False)
        plt.show()


# different to evaluate in utils, this is for inv management
class Evaluate:
    def __init__(self, env, model, n_eval_episodes=10):
        self.env = env
        self.model = model
        self.n_eval_episodes = n_eval_episodes

    def run(self, filepath=""):
        """Gets data from the model interacting with the environment

        Args:
            env (_type_): _description_
            n_eval_episodes (_type_): _description_
        """
        raw_rewards = np.zeros((self.n_eval_episodes, self.env.num_periods))
        obs = self.env.reset()

        # Here one may pull data listed above in the nomencalture section by adding the respective data name to the list below

        df_names = [
            "I",
            "T",
            "R",
            "D",
            "S",
            "B",
            "LS",
            "P",
        ]  # Initialising DataFrame names
        # ['Market Demand', 'Inventory Node Stock', 'Reorder Amount', 'Node Profit', 'Edge Quantities'] Translated from above

        for episode in range(self.n_eval_episodes):
            for timestep in range(self.env.num_periods):
                action = self.model.predict(obs)
                obs, reward, _, _ = self.env.step(action[0])
                raw_rewards[episode, timestep] = reward

            for df_name in df_names:
                df = getattr(
                    self.env, df_name
                )  # Get the DataFrame from the environment for the specific data
                df = pd.DataFrame(df)
                if not os.path.exists(
                    filepath + f"/{episode}/"
                ):  # If the directory does not exist, create it
                    os.makedirs(filepath + f"/{episode}/")

                df.to_csv(
                    filepath + f"/{episode}/{df_name}.csv"
                )  # Save the DataFrame as a csv file

            obs = self.env.reset()  # Reset the environment for the next episode
            self.env.seed_int = episode  # Select a new seed for the next episode

        return raw_rewards

    def mean_reward(self, rewards, filepath=None):
        """Gets the mean reward from the model interacting with the environment

        It saves this into an evaluations.csv file in the filepath directory

        This also includes other parameters such as: mean_rewards, std_rewards, upper_rewards,
                                                    lower_rewards, cumrewards, lower_cumrewards, upper_cumrewards}


        Args:
            env (Gym Environment): Environment to be used
            model (Stable Baselines Model): Model to be used
            n_eval_episodes (_type_): Number of evaluation episodes
            filepath (str): Path to the directory where the model is saved (determined by the environment, model and hyperparameters keys)
        """

        mean_rewards = np.mean(rewards, axis=0)
        std_rewards = np.std(rewards, axis=0)
        upper_rewards = mean_rewards + std_rewards
        lower_rewards = mean_rewards - std_rewards

        upper_cumrewards = []
        lower_cumrewards = []
        cumrewards = []

        for i, v in enumerate(mean_rewards):
            cumreward = np.sum(mean_rewards[:i])
            cumrewards.append(cumreward)
            upper_cumrewards.append(cumreward + np.sqrt(np.sum(std_rewards[:i])))
            lower_cumrewards.append(cumreward - np.sqrt(np.sum(std_rewards[:i])))

        cumrewards = np.array(cumrewards)
        upper_cumrewards = np.array(upper_cumrewards)
        lower_cumrewards = np.array(lower_cumrewards)

        eval_results = {
            "mean_rewards": mean_rewards,
            "std_rewards": std_rewards,
            "upper_rewards": upper_rewards,
            "lower_rewards": lower_rewards,
            "cumlative_rewards": cumrewards,
            "cumlative_lower_rewards": lower_cumrewards,
            "cumlative_upper_rewards": upper_cumrewards,
        }

        eval_df = pd.DataFrame(eval_results)

        if filepath == None:
            return eval_df
        elif filepath == "default":
            savepath = filepath + "/eval_results.csv"
            eval_df.to_csv(savepath, index_label="step")
            return eval_df
        elif filepath != None:  # overides the default filepath
            savepath = filepath + "/eval_results.csv"
            eval_df.to_csv(savepath, index_label="step")
            return eval_df

    def plot_episode(self, save_path):

        fig, ax = plt.subplots(nrows=4, ncols=2)
        fig.set_size_inches(8, 8)

        df_names = ["I", "T", "R", "D", "S", "B", "LS", "P"]

        data = {}

        for i, df_name in enumerate(df_names):
            df = pd.read_csv(f"{save_path}/3/{df_name}.csv")
            df.drop(columns=df.columns[0], axis=1, inplace=True)
            data[df_name] = df

        cmap = plt.get_cmap("tab20")

        ax[0, 0].plot(data["P"], linewidth=1.0, color=cmap(i))

        for i, node in enumerate(data["I"].columns):
            ax[0, 1].plot(data["I"][node], linewidth=1.0, label=node, color=cmap(i))

        for i, node in enumerate(data["T"].columns):
            ax[1, 0].plot(
                data["T"][node],
                linewidth=1.0,
                label=node,
                color=cmap(i),
            )

        for i, node in enumerate(data["R"].columns):
            ax[1, 1].plot(
                data["R"][node],
                linewidth=1.0,
                label=node,
                color=cmap(i),
            )

        for i, node in enumerate(data["D"].columns):
            ax[2, 0].plot(
                data["D"][node],
                linewidth=1.0,
                label=node,
                color=cmap(i),
            )

        for i, node in enumerate(data["S"].columns):
            ax[2, 1].plot(
                data["S"][node],
                linewidth=1.0,
                label=node,
                color=cmap(i),
            )

        for i, node in enumerate(data["B"].columns):
            ax[3, 0].plot(
                data["B"][node],
                linewidth=1.0,
                label=node,
                color=cmap(i),
            )

        for i, node in enumerate(data["LS"].columns):
            ax[3, 1].plot(
                data["LS"][node],
                linewidth=1.0,
                label=node,
                color=cmap(i),
            )

        ax[0, 1].legend(fontsize=6)
        ax[1, 0].legend(fontsize=6)
        ax[1, 1].legend(fontsize=6)

        ax[0, 0].set_xlabel("Timestep")
        ax[0, 0].set_ylabel("Profit")

        ax[0, 1].set_xlabel("Timestep")
        ax[0, 1].set_ylabel("Inventory Level")

        ax[1, 0].set_xlabel("Timestep")
        ax[1, 0].set_ylabel("Pipeline Inventory")

        ax[1, 1].set_xlabel("Timestep")
        ax[1, 1].set_ylabel("Replenishment Order")

        ax[2, 0].set_xlabel("Timestep")
        ax[2, 0].set_ylabel("Retailer Demand")

        ax[2, 1].set_xlabel("Timestep")
        ax[2, 1].set_ylabel("Units Sold")

        ax[3, 0].set_xlabel("Timestep")
        ax[3, 0].set_ylabel("Backlog")

        ax[3, 1].set_xlabel("Timestep")
        ax[3, 1].set_ylabel("Lost Sales")

        fig.tight_layout(h_pad=0.0, w_pad=0.0)
        fig.show()
        plt.show()


def train_evaluate(model_key, env_key):

    train_env = env_dictionary[env_key]
    eval_env = bullwhip_eval_env()

    exp = BullWhipExperiment(model_key, train_env, env_key)
    model, history = exp.train()
    exp.plot_train_history(history)

    bullwhip_eval = Evaluate(eval_env, model, n_eval_episodes)

    bullwhip_rewards = bullwhip_eval.run(
        f"Experiments/Experiment_data/bullwhip_data/{model_key}/{env_key}/bullwhip/"
    )

    bullwhip_df = bullwhip_eval.mean_reward(
        bullwhip_rewards,
        f"Experiments/Experiment_data/bullwhip_data/{model_key}/{env_key}/bullwhip/",
    )

    bullwhip_eval.plot_episode(
        f"Experiments/Experiment_data/bullwhip_data/{model_key}/{env_key}/bullwhip/"
    )

    eval_comparison(bullwhip_df)


def evaluate_pretrained(model_key, env_key):

    eval_env = bullwhip_eval_env()
    algorithm, _ = algorithm_dictionary[model_key][:2]
    model = algorithm.load(
        f"Experiments/Experiment_data/bullwhip_data/{model_key}/{env_key}/best_model"
    )

    bullwhip_eval = Evaluate(eval_env, model, n_eval_episodes)

    bullwhip_rewards = bullwhip_eval.run(
        f"Experiments/Experiment_data/bullwhip_data/{model_key}/{env_key}/bullwhip/"
    )

    bullwhip_df = bullwhip_eval.mean_reward(
        bullwhip_rewards,
        f"Experiments/Experiment_data/bullwhip_data/{model_key}/{env_key}bullwhip/",
    )

    bullwhip_eval.plot_episode(
        f"Experiments/Experiment_data/bullwhip_data/{model_key}/{env_key}/bullwhip/"
    )

    eval_comparison(bullwhip_df)


def main():

    print(
        "Welcome to the Bullwhip experiment! Choose an algorithm you would like to use:"
    )
    for i, (key, val) in enumerate(algorithm_dictionary.items()):
        print(f"{i} - {key}")
    choice = input("Enter a number for the algorithm you wish to use: ")
    valid_inputs = [str(i) for i in range(len(algorithm_dictionary.items()))]
    while choice not in valid_inputs:
        choice = input("Invalid input, enter a number listed above: ")

    choice = int(choice)
    algo = list(algorithm_dictionary)[choice]

    print()

    for i, (key, val) in enumerate(env_dictionary.items()):
        print(f"{i}) - {key}")
    choice = input("Choose an environmnent for training or evaluation:")
    valid_inputs = [str(i) for i in range(len(env_dictionary.items()))]
    while choice not in valid_inputs:
        choice = input("Invalid input, enter a number listed above")

    choice = int(choice)
    env = list(env_dictionary)[choice]

    while True:
        options = ["Train", "Load pre-trained", "Exit"]
        print()
        for i in range(len(options)):
            print(f"{i}. {options[i]}")

        choice = input("Enter a number to choose an action: ")
        valid_inputs = [str(i) for i in range(len(options))]

        while choice not in valid_inputs:
            choice = input("Invalid input, enter a number listed above: ")
        choice = int(choice)

        if choice == 0:
            train_evaluate(algo, env)
        elif choice == 1:
            evaluate_pretrained(algo, env)
        elif choice == 2:
            break

        print("Done!")


if __name__ == "__main__":
    main()
