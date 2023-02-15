import pandas as pd  
import tqdm 
import or_gym


from algorithms_dictionary import algorithm_dictionary # This is a dictionary with the algorithms and their parameters
# from tuned_params import tuned_params # This is a dictionary with the tuned parameters for each algorithm




def CreateEnv(env_string=None, **kwargs): 
    """ Creates an environment based on an environment string passed as argument

    Args:
        env_string (str): Environment to be used. Defaults to None.

    Returns:
        _type_: _description_
    """
    or_gym.make(env_string, seed_int=42, )
    
    env = or_gym.make(env_string, seed_int=42)
    
    env = gym.make(env_string, **kwargs)
    
    return env


def CreateModel(alg_string=None, env=None, **kwargs): 
    """ Creates a model based on an algorithm, environment and hyperparameters passed as arguments

    Args:
        alg_string (str): Algorithm to be used. Defaults to None.
        env (Gym Environment): Environment to be used. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    
    alg, pol = algorithm_dictionary[alg_string][:2]
    
    model = alg(pol, env, **kwargs)
    
    return model




if __name__ == '__main__':
    #### Code here to be executed
    
    
    CreateModel(alg_string='PPO', env=
    
    
    
    print('0')