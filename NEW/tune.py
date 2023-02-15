from stable_baselines3.common.callbacks import EvalCallback
import optuna
from hyperparam_dict import HYPERPARAMS_SAMPLER
from stable_baselines3.common.monitor import Monitor
import or_gym
from stable_baselines3 import PPO
from sb3_contrib import ARS, QRDQN, TQC, TRPO, RecurrentPPO
import argparse
import joblib
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
import pickle as pkl


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--algo', type=str, default='ppo', 
        help='name of rl algorithm to optimize')
    
    args = parser.parse_args()
    return args


ALGOS = {
    'ppo' : PPO,
    'ars' : ARS,
    'rppo' : RecurrentPPO,
    'trpo' : TRPO,
}




N_TRIALS = 10
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = int(3e5)
EVAL_FREQ = int(N_TIMESTEPS/N_EVALUATIONS)
N_EVAL_EPISODES = 5

ENV_ID = 'InvManagement-v1'

#need to change for recurrent ppo add if statement
DEFAULT_HYPERPARAMS = {
    'policy' : 'MlpPolicy',
    'env' : ENV_ID,
}

class TrialEvalCallback(EvalCallback):
    '''Callback used for evaluating and reporting a trial'''
    def __init__(
        self, 
        eval_env,
        trial,
        n_eval_episodes,
        eval_freq,
        deterministic = True,
        verbose = 0
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose
        )

        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx +=1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            #Pruned trial if need
            if self.trial.should_prune():
                self.is_prune = True
                return False
        return True




def objective(trial: optuna.Trial) -> float:

    #sample hyperparameters
    kwargs = DEFAULT_HYPERPARAMS.copy()
    kwargs.update(HYPERPARAMS_SAMPLER[args.algo](trial))

    #create model    
    model = ALGOS[args.algo](**kwargs)

    eval_env = Monitor(or_gym.make(ENV_ID))

    eval_callback = TrialEvalCallback(eval_env=eval_env, trial=trial,
                        n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ,
                        deterministic=True)

    nan_encountered = False

    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        #sometimes model generates Nan.
        print(e)
        nan_encountered = True
    finally:
        #free memory
        model.env.close()
        eval_env.close()

    #tell optimizer trial failed
    if nan_encountered:
        return float('nan')

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()
    
    return eval_callback.last_mean_reward


if __name__ == '__main__':
    args = parse_args()

    study_path = f'tuning/{args.algo}'

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3) 

    study = optuna.create_study(
        study_name='ppo_study',
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
        direction='maximize',
    )

    try:
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    except KeyboardInterrupt:
        pass

    print(f'Number of finished trials: {len(study.trials)}')

    trial = study.best_trial
    print(f'Best trial: {trial}')
    print(f'Value: {trial.value}')


    print('Params:')
    for key, value in trial.params.items():
        print(f' {key}: {value}')



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
